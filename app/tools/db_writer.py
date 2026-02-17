"""
Database writer: persists the final order result into Supabase tables.
Executes inserts sequentially (prescription → order → items → customer update
→ job update → notification → internal note).
Each step is wrapped in try/except so partial failures don't lose work.

Sale-tag aware:
  - Includes order_type in the order INSERT payload
  - For venta_directa: uses suggested_status instead of 'borrador'
  - Adjusts notification messages for direct sales
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.models.job import AIOrderJob
from app.models.order_draft import FinalOrderResult
from app.tools.supabase_client import get_supabase

logger = logging.getLogger(__name__)


def persist_order_result(job: AIOrderJob, result: FinalOrderResult) -> str | None:
    """
    Write the complete order draft to the database.
    Returns the created order ID, or None if the order insert itself failed.
    """
    sb = get_supabase()
    order_id: str | None = None
    prescription_id: str | None = None
    order_number: int | None = None
    errors: list[str] = []

    is_venta_directa = result.order_type == "venta_directa"

    # ── 1. Insert prescription (if available and NOT venta_directa) ──
    if not is_venta_directa and result.prescription and result.prescription.rx_data:
        try:
            rx_payload: dict[str, Any] = {
                "customer_id": result.prescription.customer_id or job.customer_id,
                "rx_data": result.prescription.rx_data.model_dump(exclude_none=True),
            }
            if result.prescription.original_image_url:
                rx_payload["original_image_url"] = result.prescription.original_image_url
            if result.prescription.ai_extraction_metadata:
                rx_payload["ai_extraction_metadata"] = (
                    result.prescription.ai_extraction_metadata.model_dump(exclude_none=True)
                )

            rx_resp = sb.table("prescriptions").insert(rx_payload).execute()
            if rx_resp.data:
                prescription_id = rx_resp.data[0]["id"]
                logger.info("Inserted prescription %s", prescription_id)
        except Exception as exc:
            msg = f"Error inserting prescription: {exc}"
            logger.error(msg, exc_info=True)
            errors.append(msg)

    # ── 2. Insert order ───────────────────────────────────────
    try:
        # Determine the status for the order
        order_status = result.order_draft.status
        if is_venta_directa and result.suggested_status:
            order_status = result.suggested_status

        order_payload: dict[str, Any] = {
            "customer_id": result.order_draft.customer_id or job.customer_id,
            "sede_id": result.order_draft.sede_id or job.sede_id,
            "seller_id": result.order_draft.seller_id or job.requested_by,
            "status": order_status,
            "order_type": result.order_draft.order_type or "optico",
            "total_amount": result.order_draft.total_amount or 0,
            "balance_due": result.order_draft.balance_due or 0,
            "payment_status": result.order_draft.payment_status or "pendiente",
        }
        if result.order_draft.lab_id and not is_venta_directa:
            order_payload["lab_id"] = result.order_draft.lab_id
        if result.order_draft.promised_date:
            order_payload["promised_date"] = result.order_draft.promised_date

        order_resp = sb.table("orders").insert(order_payload).execute()
        if order_resp.data:
            order_id = order_resp.data[0]["id"]
            order_number = order_resp.data[0].get("order_number")
            logger.info("Inserted order %s (number: %s, type: %s)", order_id, order_number, result.order_type)
    except Exception as exc:
        msg = f"Error inserting order: {exc}"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        # If we can't create the order, fail the job
        _fail_job(sb, job, msg)
        return None

    # ── 3. Insert order items ─────────────────────────────────
    for idx, item in enumerate(result.items):
        try:
            item_payload: dict[str, Any] = {
                "order_id": order_id,
                "description": item.description or f"Item {idx + 1}",
                "quantity": item.quantity or 1,
                "unit_price": item.unit_price or 0,
            }
            if item.lens_catalog_id:
                item_payload["lens_catalog_id"] = item.lens_catalog_id
            if item.lens_lab_cost is not None:
                item_payload["lens_lab_cost"] = item.lens_lab_cost
            if item.product_id:
                item_payload["product_id"] = item.product_id
            # Link prescription to lens items (only for optico orders)
            if prescription_id and not is_venta_directa and item.description and "lente" in (item.description or "").lower():
                item_payload["prescription_id"] = prescription_id

            sb.table("order_items").insert(item_payload).execute()
            logger.info("Inserted order_item %d for order %s", idx, order_id)
        except Exception as exc:
            msg = f"Error inserting order_item {idx}: {exc}"
            logger.error(msg, exc_info=True)
            errors.append(msg)

    # ── 4. Update customer (if updates available) ─────────────
    if result.customer_updates:
        try:
            updates: dict[str, Any] = {}
            cu = result.customer_updates
            if cu.email:
                updates["email"] = cu.email
            if cu.document_id:
                updates["document_id"] = cu.document_id
            if cu.city:
                updates["city"] = cu.city
            if cu.phone:
                updates["phone"] = cu.phone
            if cu.address:
                updates["address"] = cu.address

            if updates:
                sb.table("customers").update(updates).eq("id", job.customer_id).execute()
                logger.info("Updated customer %s with %s", job.customer_id, list(updates.keys()))
        except Exception as exc:
            msg = f"Error updating customer: {exc}"
            logger.error(msg, exc_info=True)
            errors.append(msg)

    # ── 5. Update job status ──────────────────────────────────
    try:
        # Add any DB write errors to the result warnings
        result_dict = result.model_dump(exclude_none=True)
        if errors:
            result_dict.setdefault("warnings", []).extend(errors)

        sb.table("ai_order_jobs").update({
            "status": "completed",
            "result": result_dict,
            "order_id": order_id,
            "completed_at": "now()",
        }).eq("id", job.id).execute()
        logger.info("Job %s completed → order %s", job.id, order_id)
    except Exception as exc:
        logger.error("Error updating job %s: %s", job.id, exc, exc_info=True)

    # ── 6. Insert notification for the requesting user ────────
    try:
        display_number = f"#{order_number}" if order_number else "(sin número)"
        completeness_label = result.completeness or "minimo"
        severity = "info" if completeness_label == "completo" else "warning"

        if is_venta_directa:
            title = f"Venta directa creada {display_number}"
            message = (
                f"Venta directa {display_number} creada por IA. "
                f"Estado: {completeness_label}. "
                f"Total: ${result.order_draft.total_amount:,.0f} COP."
            )
        else:
            title = f"Pedido IA creado {display_number}"
            message = (
                f"Pedido borrador {display_number} creado por IA. "
                f"Estado: {completeness_label}. Pendiente revisión por logística."
            )

        sb.table("notifications").insert({
            "user_id": job.requested_by,
            "type": "ai_order",
            "title": title,
            "message": message,
            "severity": severity,
            "metadata": {
                "order_id": order_id,
                "order_type": result.order_type,
                "completeness": completeness_label,
                "needs_manual_review": result.needs_manual_review,
                "warnings_count": len(result.warnings),
            },
            "link_to": "/admin/verification-queue",
        }).execute()
        logger.info("Notification sent to user %s", job.requested_by)
    except Exception as exc:
        logger.error("Error inserting notification: %s", exc, exc_info=True)

    # ── 7. Insert internal note in conversation ───────────────
    try:
        display_number = f"#{order_number}" if order_number else ""
        warnings_text = ""
        if result.warnings:
            warnings_text = "\n⚠️ " + "\n⚠️ ".join(result.warnings[:5])

        if is_venta_directa:
            note_content = (
                f"🤖 Venta directa {display_number} creada por IA.\n"
                f"Estado: {result.completeness}.\n"
                f"Total: ${result.order_draft.total_amount:,.0f} COP"
                f"{warnings_text}"
            )
        else:
            note_content = (
                f"🤖 Pedido borrador {display_number} creado por IA.\n"
                f"Estado: {result.completeness}.\n"
                f"Total: ${result.order_draft.total_amount:,.0f} COP"
                f"{warnings_text}"
            )

        sb.table("messages").insert({
            "conversation_id": job.conversation_id,
            "sender_type": "system",
            "is_internal": True,
            "content": note_content,
            "message_type": "text",
        }).execute()
        logger.info("Internal note inserted for conversation %s", job.conversation_id)
    except Exception as exc:
        logger.error("Error inserting internal note: %s", exc, exc_info=True)

    return order_id


def _fail_job(sb: Any, job: AIOrderJob, error_message: str) -> None:
    """Mark a job as failed and insert an error note in the conversation."""
    try:
        sb.table("ai_order_jobs").update({
            "status": "failed",
            "error_message": error_message,
            "completed_at": "now()",
        }).eq("id", job.id).execute()
    except Exception:
        logger.error("Could not mark job %s as failed", job.id, exc_info=True)

    try:
        sb.table("messages").insert({
            "conversation_id": job.conversation_id,
            "sender_type": "system",
            "is_internal": True,
            "content": f"⚠️ Error al procesar pedido IA: {error_message}",
            "message_type": "text",
        }).execute()
    except Exception:
        logger.error("Could not insert error note for job %s", job.id, exc_info=True)

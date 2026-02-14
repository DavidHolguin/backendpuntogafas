"""
Agent 4: Order Builder — assembles the final order draft from all
previous agent outputs. Pure Python logic, no LLM calls.

Merges payment data from remission (vision) + conversation into
a single PaymentSuggestion. Priority: remission > conversation.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.models.extraction import CatalogOutput, ConversationOutput, VisionOutput
from app.models.job import AIOrderJob
from app.models.order_draft import (
    FinalOrderResult,
    OrderDraftHeader,
    OrderDraftItem,
    PrescriptionInsert,
)
from app.models.prescription import AiExtractionMetadata, PaymentSuggestion

logger = logging.getLogger(__name__)


def _build_payment_suggestion(
    vision: VisionOutput,
    conversation: ConversationOutput,
) -> PaymentSuggestion | None:
    """
    Merge payment info from remission + conversation into a single suggestion.
    Priority: remission > conversation (remission is a signed document).
    amount_reference is INFORMATIONAL ONLY — never overwrites catalog total.
    """
    remission = vision.remissions[0] if vision.remissions else None
    conv_payment = conversation.payment_mentions[0] if conversation.payment_mentions else None

    # Source 1: Remission (highest priority)
    if remission and remission.payment_method:
        return PaymentSuggestion(
            method=remission.payment_method,
            type=remission.payment_type or "total",
            amount_reference=remission.payment_amount or remission.total_amount,
            has_proof=remission.has_proof,
            source="remission",
            proof_url=None,
        )

    # Source 2: Conversation mentions
    if conv_payment and conv_payment.method:
        return PaymentSuggestion(
            method=conv_payment.method,
            type=conv_payment.type or "total",
            amount_reference=conv_payment.amount,
            has_proof=conv_payment.has_proof,
            source=conv_payment.source,
            proof_url=conv_payment.proof_url,
        )

    # No payment info found — check if remission exists but without method
    if remission and remission.payment_info:
        return PaymentSuggestion(
            method=None,
            type="total",
            amount_reference=remission.total_amount,
            has_proof=False,
            source="remission",
            proof_url=None,
        )

    return None


def run_order_builder(
    job: AIOrderJob,
    vision: VisionOutput,
    conversation: ConversationOutput,
    catalog: CatalogOutput,
    agent_errors: dict[str, str] | None = None,
    processing_start: float | None = None,
) -> FinalOrderResult:
    """
    Assemble the final order draft from all agent outputs.
    ALWAYS returns a valid FinalOrderResult — never raises.
    """
    warnings: list[str] = []
    items: list[OrderDraftItem] = []

    # ── Build order items from catalog matches ────────────────
    for mi in catalog.matched_items:
        subtotal = (mi.unit_price or 0) * (mi.quantity or 1)
        items.append(OrderDraftItem(
            description=mi.description,
            quantity=mi.quantity or 1,
            unit_price=mi.unit_price or 0,
            subtotal=subtotal,
            lens_catalog_id=mi.lens_catalog_id,
            lens_lab_cost=mi.lab_cost,
            product_id=mi.product_id,
            needs_manual_selection=mi.needs_manual_selection,
        ))

    # ── Calculate totals (from catalog items ONLY — precio es sagrado) ──
    total_amount = sum(item.subtotal for item in items)
    balance_due = total_amount

    # ── Remission data (REFERENCE ONLY) ──────────────────────
    remission = vision.remissions[0] if vision.remissions else None

    # NOTE: We intentionally do NOT use remission.total_amount to set the
    # order total. The total comes exclusively from catalog item prices.
    # The remission amount is stored as reference for the logistic team.

    # ── Clinical history (use first found) ────────────────────
    clinical_history = vision.clinical_histories[0] if vision.clinical_histories else None

    # ── Frames ────────────────────────────────────────────────
    frames = vision.frames

    # ── Payment suggestion (merged from all sources) ──────────
    payment_suggestion = _build_payment_suggestion(vision, conversation)

    # ── Build prescription (use first found) ──────────────────
    prescription: PrescriptionInsert | None = None
    if vision.prescriptions_found:
        best_rx = vision.prescriptions_found[0]
        prescription = PrescriptionInsert(
            customer_id=job.customer_id,
            rx_data=best_rx.rx_data,
            original_image_url=best_rx.image_url,
            ai_extraction_metadata=AiExtractionMetadata(
                confidence=best_rx.confidence,
                model="gemini-2.0-flash",
                warnings=best_rx.warnings,
                extracted_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        if best_rx.warnings:
            warnings.extend([f"Fórmula: {w}" for w in best_rx.warnings])

    # ── Determine lab_id ──────────────────────────────────────
    lab_id = catalog.suggested_lab_id

    # ── Promised date from conversation ───────────────────────
    promised_date = conversation.promised_date_hint

    # ── Consolidate warnings ──────────────────────────────────
    warnings.extend(conversation.warnings)
    warnings.extend(catalog.warnings)

    if vision.error:
        warnings.append(f"Agente Visión: {vision.error}")
    if conversation.error:
        warnings.append(f"Agente Conversación: {conversation.error}")
    if catalog.error:
        warnings.append(f"Agente Catálogo: {catalog.error}")

    if not items:
        warnings.append("No se identificaron productos — pedido vacío requiere revisión manual")

    if total_amount == 0 and items:
        warnings.append("Total $0 — precios pendientes de asignar por logística")

    if not prescription and job.payload.media_urls:
        warnings.append("Se enviaron imágenes pero no se extrajo fórmula óptica")

    any_manual = any(i.needs_manual_selection for i in items)
    if any_manual:
        warnings.append("Uno o más items requieren selección manual por logística")

    if remission and remission.observations:
        obs = remission.observations.upper()
        if "URGENTE" in obs:
            warnings.insert(0, f"🔴 URGENTE — observación de remisión: {remission.observations}")

    # Warn if remission total differs significantly from catalog total
    if remission and remission.total_amount and total_amount > 0:
        diff = abs(remission.total_amount - total_amount)
        if diff > 1000:  # More than $1.000 COP difference
            warnings.append(
                f"⚠️ Remisión dice ${remission.total_amount:,.0f} pero "
                f"catálogo calcula ${total_amount:,.0f} — verificar"
            )

    if payment_suggestion and payment_suggestion.has_proof:
        warnings.append("📎 Comprobante de pago detectado — requiere verificación")

    # ── Determine completeness ────────────────────────────────
    has_items = len(items) > 0
    has_prices = total_amount > 0
    has_prescription = prescription is not None
    no_manual = not any_manual

    if has_items and has_prices and no_manual:
        if has_prescription or not job.payload.media_urls:
            completeness = "completo"
        else:
            completeness = "parcial"
    elif has_items:
        completeness = "parcial"
    else:
        completeness = "minimo"

    needs_manual_review = completeness != "completo"

    # ── Processing time ───────────────────────────────────────
    processing_time_ms = 0
    if processing_start:
        processing_time_ms = int((time.time() - processing_start) * 1000)

    # ── Assemble final result ─────────────────────────────────
    result = FinalOrderResult(
        order_draft=OrderDraftHeader(
            customer_id=job.customer_id,
            sede_id=job.sede_id,
            seller_id=job.requested_by,
            status="borrador",
            total_amount=total_amount,
            balance_due=balance_due,
            payment_status="pendiente",
            lab_id=lab_id,
            promised_date=promised_date,
        ),
        items=items,
        prescription=prescription,
        remission=remission,
        clinical_history=clinical_history,
        frames=frames,
        image_classifications=vision.image_classifications,
        payment_suggestion=payment_suggestion,
        customer_updates=conversation.customer_updates,
        completeness=completeness,
        warnings=warnings,
        needs_manual_review=needs_manual_review,
        processing_time_ms=processing_time_ms,
        agent_errors=agent_errors or {},
    )

    logger.info(
        "Order built: %d items, total=$%.0f, completeness=%s, "
        "remission=%s, clinical=%s, frames=%d, payment=%s, warnings=%d",
        len(items), total_amount, completeness,
        "yes" if remission else "no",
        "yes" if clinical_history else "no",
        len(frames),
        payment_suggestion.method if payment_suggestion else "none",
        len(warnings),
    )

    return result

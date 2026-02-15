"""
Shipping Guide Crew — Orchestrates the 3-agent sequential pipeline.

1. Vision Agent → extract data from image
2. Validation Agent → check duplicates, confidence, resolve carrier
3. Matching Agent → fuzzy match to listo_entrega orders

Post-processing: write shipping_guide + notifications.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.agents.shipping.matching_agent import match_order
from app.agents.shipping.validation_agent import validate_guide
from app.agents.shipping.vision_agent import extract_guide_data
from app.models.shipping import (
    ExtractedGuideData,
    ShippingExtractRequest,
    ShippingExtractResponse,
)
from app.tools.shipping_tools import (
    call_order_notify,
    create_shipping_guide,
    send_orphan_notification,
)
from app.tools.supabase_client import get_supabase

logger = logging.getLogger(__name__)


def run_shipping_crew(request: ShippingExtractRequest) -> ShippingExtractResponse:
    """
    Execute the full shipping guide extraction pipeline.

    Args:
        request: The incoming extraction request from whatsapp-media-processor

    Returns:
        ShippingExtractResponse with guide_id, order match, etc.
    """
    start = time.time()

    logger.info("=" * 60)
    logger.info("SHIPPING GUIDE CREW START")
    logger.info("Image: %s", request.image_url[:80])
    logger.info("Carrier ID: %s", request.carrier_company_id)
    logger.info("=" * 60)

    # ── Agent 1: Vision Extraction ────────────────────────────

    logger.info("=== AGENT 1: Vision Extractor ===")

    # Build carrier context if we have a carrier_company_id
    carrier_context = ""
    if request.carrier_company_id:
        try:
            sb = get_supabase()
            resp = (
                sb.table("carrier_companies")
                .select("name,code,tracking_url_template")
                .eq("id", request.carrier_company_id)
                .limit(1)
                .execute()
            )
            if resp.data:
                c = resp.data[0]
                carrier_context = (
                    f"\nEsta imagen proviene de la transportadora: "
                    f"{c['name']} ({c['code']}). "
                    f"URL de rastreo base: {c.get('tracking_url_template') or 'desconocida'}."
                )
        except Exception as exc:
            logger.warning("Could not fetch carrier context: %s", exc)

    try:
        extracted = extract_guide_data(
            image_url=request.image_url,
            carrier_context=carrier_context,
        )
        logger.info(
            "Vision result: tracking=%s, carrier=%s, recipient=%s, confidence=%.2f",
            extracted.tracking_code,
            extracted.carrier_name,
            extracted.recipient_name,
            extracted.confidence,
        )
    except Exception as exc:
        logger.error("Vision extractor failed: %s", exc, exc_info=True)
        return ShippingExtractResponse(
            success=False,
            error=f"Vision extraction failed: {exc}",
        )

    # ── Agent 2: Validation ───────────────────────────────────

    logger.info("=== AGENT 2: Validation ===")

    validation = validate_guide(
        tracking_code=extracted.tracking_code,
        carrier_name=extracted.carrier_name,
        confidence=extracted.confidence,
        carrier_company_id=request.carrier_company_id,
    )

    if not validation["is_valid"]:
        if validation["is_duplicate"]:
            return ShippingExtractResponse(
                success=True,
                duplicate=True,
                guide_id=validation["duplicate_guide_id"],
                extracted=extracted,
            )
        return ShippingExtractResponse(
            success=False,
            error=validation["rejection_reason"],
            extracted=extracted,
        )

    carrier_company_id = validation["carrier_company_id"]

    # ── Agent 3: Order Matching ───────────────────────────────

    logger.info("=== AGENT 3: Order Matching ===")

    match_result = match_order(
        recipient_name=extracted.recipient_name,
        recipient_address=extracted.recipient_address,
        recipient_city=extracted.recipient_city,
    )

    matched_order_id = match_result["matched_order_id"]
    match_score = match_result["match_score"]

    # ── Post-Processing: Create Guide + Notifications ─────────

    logger.info("=== POST-PROCESSING: Write Results ===")

    guide_data = {
        "order_id": matched_order_id,
        "carrier_name": extracted.carrier_name or "Desconocida",
        "tracking_code": extracted.tracking_code,
        "tracking_url": extracted.tracking_url,
        "carrier_company_id": carrier_company_id,
        "source_type": "ai_extracted",
        "source_message_id": request.message_id,
        "status": "in_transit" if matched_order_id else "orphan",
        "recipient_name": extracted.recipient_name,
        "recipient_address": extracted.recipient_address,
        "recipient_city": extracted.recipient_city,
        "source_extracted_data": {
            "tracking_code": extracted.tracking_code,
            "carrier_name": extracted.carrier_name,
            "tracking_url": extracted.tracking_url,
            "recipient_name": extracted.recipient_name,
            "recipient_address": extracted.recipient_address,
            "recipient_city": extracted.recipient_city,
            "confidence": extracted.confidence,
            "match_score": match_score,
            "matched_at": (
                datetime.now(timezone.utc).isoformat()
                if matched_order_id
                else None
            ),
            "image_url": request.image_url,
            "processing_backend": "python_crewai",
        },
    }

    try:
        guide_record = create_shipping_guide(guide_data)
        guide_id = guide_record["id"] if guide_record else None
    except Exception as exc:
        logger.error("Failed to create shipping guide: %s", exc, exc_info=True)
        return ShippingExtractResponse(
            success=False,
            error=f"Database insert failed: {exc}",
            extracted=extracted,
        )

    if not guide_id:
        return ShippingExtractResponse(
            success=False,
            error="Guide created but no ID returned",
            extracted=extracted,
        )

    logger.info(
        "Guide created: %s (%s)",
        guide_id,
        "matched" if matched_order_id else "orphan",
    )

    # ── Notifications ─────────────────────────────────────────

    if matched_order_id:
        call_order_notify(
            order_id=matched_order_id,
            carrier_name=extracted.carrier_name,
            tracking_code=extracted.tracking_code,
            tracking_url=extracted.tracking_url,
        )
    else:
        send_orphan_notification(
            guide_id=guide_id,
            tracking_code=extracted.tracking_code or "",
            carrier_name=extracted.carrier_name,
            recipient_name=extracted.recipient_name,
            recipient_address=extracted.recipient_address,
            recipient_city=extracted.recipient_city,
        )

    elapsed = time.time() - start
    logger.info(
        "SHIPPING GUIDE CREW COMPLETE in %.1fs — guide=%s, order=%s, score=%.4f",
        elapsed, guide_id, matched_order_id, match_score,
    )

    return ShippingExtractResponse(
        success=True,
        guide_id=guide_id,
        order_id=matched_order_id,
        is_orphan=not matched_order_id,
        match_score=match_score,
        extracted=extracted,
    )

"""
Pipeline orchestrator — runs the 4-agent pipeline sequentially.
Each agent is wrapped in try/except, so a failure in one agent
does NOT stop the pipeline. Fallback outputs are used instead.

Sale-tag aware:
  - Passes internal_notes to vision extractor for tag-based classification
  - Propagates suggested_order_type through the pipeline
"""

from __future__ import annotations

import logging
import time

from app.agents.catalog_matcher import run_catalog_matcher
from app.agents.conversation_analyzer import run_conversation_analyzer
from app.agents.order_builder import run_order_builder
from app.agents.vision_extractor import run_vision_extractor
from app.models.extraction import CatalogOutput, ConversationOutput, VisionOutput
from app.models.job import AIOrderJob
from app.models.order_draft import FinalOrderResult

logger = logging.getLogger(__name__)


def run_pipeline(job: AIOrderJob) -> FinalOrderResult:
    """
    Execute the full extraction pipeline for an AI order job.
    Tolerant to failures in any agent — always returns a result.
    """
    processing_start = time.time()
    agent_errors: dict[str, str] = {}
    payload = job.payload

    # ── Agent 1: Vision Extractor ─────────────────────────────
    logger.info("=== AGENT 1: Vision Extractor ===")
    vision = VisionOutput()
    try:
        vision = run_vision_extractor(
            media_urls=payload.media_urls,
            internal_notes=payload.internal_notes,
        )
        logger.info(
            "Vision: %d prescriptions, %d remissions, %d frames, %d classifications",
            len(vision.prescriptions_found),
            len(vision.remissions),
            len(vision.frames),
            len(vision.image_classifications),
        )
    except Exception as exc:
        error_msg = f"Vision extractor fallo: {exc}"
        logger.error(error_msg, exc_info=True)
        agent_errors["vision_extractor"] = error_msg
        vision = VisionOutput(error=error_msg)

    # ── Agent 2: Conversation Analyzer ────────────────────────
    logger.info("=== AGENT 2: Conversation Analyzer ===")
    conversation = ConversationOutput()
    try:
        conversation = run_conversation_analyzer(
            messages=payload.messages,
            internal_notes=payload.internal_notes,
            instructions=payload.instructions,
        )
        logger.info(
            "Conversation: %d items, %d payment_mentions, urgency=%s, order_type=%s",
            len(conversation.items_requested),
            len(conversation.payment_mentions),
            conversation.urgency,
            conversation.suggested_order_type,
        )
    except Exception as exc:
        error_msg = f"Conversation analyzer fallo: {exc}"
        logger.error(error_msg, exc_info=True)
        agent_errors["conversation_analyzer"] = error_msg
        conversation = ConversationOutput(
            error=error_msg,
            warnings=["El analizador de conversación falló — pedido puede estar incompleto"],
        )

    # ── Agent 3: Catalog Matcher ──────────────────────────────
    logger.info("=== AGENT 3: Catalog Matcher ===")
    catalog = CatalogOutput()
    try:
        catalog = run_catalog_matcher(conversation, vision)
        logger.info(
            "Catalog: %d matched items, lab=%s",
            len(catalog.matched_items),
            catalog.suggested_lab_id,
        )
    except Exception as exc:
        error_msg = f"Catalog matcher fallo: {exc}"
        logger.error(error_msg, exc_info=True)
        agent_errors["catalog_matcher"] = error_msg
        catalog = CatalogOutput(
            error=error_msg,
            warnings=["El matcher de catálogo falló — items sin precios ni IDs"],
        )

    # ── Agent 4: Order Builder ────────────────────────────────
    logger.info("=== AGENT 4: Order Builder ===")
    try:
        result = run_order_builder(
            job=job,
            vision=vision,
            conversation=conversation,
            catalog=catalog,
            agent_errors=agent_errors,
            processing_start=processing_start,
        )
    except Exception as exc:
        error_msg = f"Order builder fallo: {exc}"
        logger.error(error_msg, exc_info=True)
        agent_errors["order_builder"] = error_msg

        # Last resort: build a minimal result
        result = FinalOrderResult(
            completeness="minimo",
            needs_manual_review=True,
            warnings=[
                "El constructor de pedido falló — pedido mínimo creado",
                error_msg,
            ],
            agent_errors=agent_errors,
            processing_time_ms=int((time.time() - processing_start) * 1000),
        )
        # Ensure the header at least has the job IDs
        result.order_draft.customer_id = job.customer_id
        result.order_draft.sede_id = job.sede_id
        result.order_draft.seller_id = job.requested_by

    elapsed = time.time() - processing_start
    logger.info(
        "Pipeline complete in %.1fs: completeness=%s, items=%d, order_type=%s, warnings=%d",
        elapsed, result.completeness, len(result.items),
        result.order_type, len(result.warnings),
    )

    return result

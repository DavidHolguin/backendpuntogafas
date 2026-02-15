"""
Validation Agent — Checks for duplicate guides and validates format.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.tools.shipping_tools import check_duplicate_guide, lookup_carrier_company

logger = logging.getLogger(__name__)


def validate_guide(
    tracking_code: str | None,
    carrier_name: str | None,
    confidence: float,
    carrier_company_id: str | None = None,
) -> dict:
    """
    Validate the extracted guide data.

    Returns:
        dict with keys:
          - is_valid: bool
          - is_duplicate: bool
          - duplicate_guide_id: str | None
          - carrier_company_id: str | None (resolved from carrier_name if not provided)
          - carrier_info: dict | None
          - rejection_reason: str | None
    """
    result = {
        "is_valid": True,
        "is_duplicate": False,
        "duplicate_guide_id": None,
        "carrier_company_id": carrier_company_id,
        "carrier_info": None,
        "rejection_reason": None,
    }

    # Check 1: tracking_code must exist
    if not tracking_code:
        result["is_valid"] = False
        result["rejection_reason"] = "No tracking code extracted"
        logger.warning("Validation failed: no tracking_code")
        return result

    # Check 2: confidence threshold
    if confidence < 0.3:
        result["is_valid"] = False
        result["rejection_reason"] = f"Low confidence: {confidence:.2f} (min 0.3)"
        logger.warning("Validation failed: confidence %.2f < 0.3", confidence)
        return result

    # Check 3: duplicate check
    try:
        dup_result = json.loads(
            check_duplicate_guide.run(tracking_code)
        )
        if dup_result.get("is_duplicate"):
            result["is_valid"] = False
            result["is_duplicate"] = True
            result["duplicate_guide_id"] = dup_result.get("guide_id")
            result["rejection_reason"] = f"Duplicate guide: {dup_result.get('guide_id')}"
            logger.info("Duplicate guide found: %s", tracking_code)
            return result
    except Exception as exc:
        logger.error("Error checking duplicates: %s", exc)
        # Non-fatal, continue

    # Check 4: resolve carrier_company_id from carrier_name
    if not carrier_company_id and carrier_name:
        try:
            carrier_result = json.loads(
                lookup_carrier_company.run(carrier_name)
            )
            if carrier_result:
                result["carrier_company_id"] = carrier_result.get("id")
                result["carrier_info"] = carrier_result
                logger.info(
                    "Resolved carrier: %s → %s",
                    carrier_name, carrier_result.get("code"),
                )
        except Exception as exc:
            logger.error("Error looking up carrier: %s", exc)

    logger.info(
        "Validation passed: tracking_code=%s, confidence=%.2f",
        tracking_code, confidence,
    )
    return result

"""
Matching Agent — Fuzzy matches extracted recipient data to listo_entrega orders.
Uses Jaccard similarity on name + address with city as a tiebreaker.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.tools.shipping_tools import (
    fuzzy_match_order,
    query_orders_listo_entrega,
)

logger = logging.getLogger(__name__)


def match_order(
    recipient_name: str | None,
    recipient_address: str | None,
    recipient_city: str | None,
) -> dict:
    """
    Match extracted recipient data to an order with status='listo_entrega'.

    Returns:
        dict with keys:
          - matched_order_id: str | None
          - match_score: float
    """
    if not recipient_name and not recipient_address:
        logger.warning("No recipient data to match")
        return {"matched_order_id": None, "match_score": 0.0}

    # Step 1: Query candidates
    try:
        orders_json = query_orders_listo_entrega.run()
        orders = json.loads(orders_json)

        if not orders:
            logger.info("No orders with status='listo_entrega' found")
            return {"matched_order_id": None, "match_score": 0.0}

        logger.info("Found %d candidate orders for matching", len(orders))

    except Exception as exc:
        logger.error("Error querying orders: %s", exc)
        return {"matched_order_id": None, "match_score": 0.0}

    # Step 2: Fuzzy match
    try:
        match_result = json.loads(
            fuzzy_match_order.run(
                recipient_name=recipient_name or "",
                recipient_address=recipient_address or "",
                recipient_city=recipient_city or "",
                orders_json=orders_json,
            )
        )

        order_id = match_result.get("best_order_id")
        score = match_result.get("match_score", 0.0)

        if order_id:
            logger.info(
                "Matched order %s with score %.4f", order_id, score
            )
        else:
            logger.info("No match found (best score: %.4f)", score)

        return {
            "matched_order_id": order_id,
            "match_score": score,
        }

    except Exception as exc:
        logger.error("Error in fuzzy matching: %s", exc)
        return {"matched_order_id": None, "match_score": 0.0}

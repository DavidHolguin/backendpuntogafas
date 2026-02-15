"""
Shipping Guide Tools — Supabase queries and notification helpers.
Used by CrewAI agents as @tool functions.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Optional

from crewai.tools import tool

from app.tools.supabase_client import get_supabase

logger = logging.getLogger(__name__)


# ── Text normalization & fuzzy matching ───────────────────────


def normalize(text: str) -> str:
    """Remove accents, lowercase, strip."""
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    return text.lower().strip()


def tokenize(text: str) -> list[str]:
    """Split normalized text into words > 1 char."""
    return [t for t in normalize(text).split() if len(t) > 1]


def jaccard_similarity(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two token lists."""
    set_a, set_b = set(a), set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ── CrewAI Tool Functions ─────────────────────────────────────


@tool("check_duplicate_guide")
def check_duplicate_guide(tracking_code: str) -> str:
    """Check if a shipping guide with this tracking code already exists.
    Returns JSON with 'is_duplicate' and optional 'guide_id'."""
    sb = get_supabase()
    try:
        resp = (
            sb.table("shipping_guides")
            .select("id")
            .eq("tracking_code", tracking_code)
            .limit(1)
            .execute()
        )
        if resp.data:
            return json.dumps({"is_duplicate": True, "guide_id": resp.data[0]["id"]})
        return json.dumps({"is_duplicate": False})
    except Exception as exc:
        logger.error("check_duplicate_guide error: %s", exc)
        return json.dumps({"is_duplicate": False, "error": str(exc)})


@tool("lookup_carrier_company")
def lookup_carrier_company(carrier_name: str) -> str:
    """Look up a carrier company by name/code. Returns JSON with carrier info or null."""
    sb = get_supabase()
    try:
        resp = (
            sb.table("carrier_companies")
            .select("id,code,name,tracking_url_template")
            .eq("is_active", "true")
            .execute()
        )
        normalized_input = normalize(carrier_name)
        for c in resp.data:
            if (
                normalized_input in normalize(c["code"])
                or normalized_input in normalize(c["name"])
                or normalize(c["code"]) in normalized_input
                or normalize(c["name"]) in normalized_input
            ):
                return json.dumps(c)
        return json.dumps(None)
    except Exception as exc:
        logger.error("lookup_carrier_company error: %s", exc)
        return json.dumps(None)


@tool("query_orders_listo_entrega")
def query_orders_listo_entrega() -> str:
    """Query orders with status='listo_entrega' for matching.
    Returns JSON array of orders with customer info."""
    sb = get_supabase()
    try:
        resp = (
            sb.table("orders")
            .select("id,order_number,customer_id,customers(full_name,address,city)")
            .eq("status", "listo_entrega")
            .order("status_changed_at", desc=True)
            .limit(50)
            .execute()
        )
        return json.dumps(resp.data, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("query_orders_listo_entrega error: %s", exc)
        return json.dumps([])


@tool("fuzzy_match_order")
def fuzzy_match_order(
    recipient_name: str,
    recipient_address: str,
    recipient_city: str,
    orders_json: str,
) -> str:
    """Perform Jaccard fuzzy matching of extracted recipient data against orders.
    Returns JSON with best_order_id and match_score."""
    try:
        orders = json.loads(orders_json)
    except Exception:
        return json.dumps({"best_order_id": None, "match_score": 0.0})

    extracted_name_tokens = tokenize(recipient_name or "")
    extracted_address_tokens = tokenize(recipient_address or "")
    extracted_city = normalize(recipient_city or "")

    best_score = 0.0
    best_order_id: Optional[str] = None

    for order in orders:
        # Handle PostgREST embedded relation (customers is an object)
        customer = order.get("customers") or {}
        if isinstance(customer, list):
            customer = customer[0] if customer else {}

        full_name = customer.get("full_name", "")
        address = customer.get("address", "")
        city = customer.get("city", "")

        # Name similarity (most important)
        name_sim = jaccard_similarity(
            extracted_name_tokens, tokenize(full_name)
        )

        # Address similarity
        addr_sim = jaccard_similarity(
            extracted_address_tokens, tokenize(address)
        )

        # City match (contains)
        cust_city = normalize(city)
        city_match = 1.0 if (
            extracted_city and cust_city and
            (extracted_city in cust_city or cust_city in extracted_city)
        ) else 0.0

        # Combined score
        score = name_sim * 0.6 + addr_sim * 0.25 + city_match * 0.15

        if score > best_score and name_sim >= 0.5:
            best_score = score
            best_order_id = order["id"]

    if best_order_id and best_score >= 0.4:
        return json.dumps({
            "best_order_id": best_order_id,
            "match_score": round(best_score, 4),
        })

    return json.dumps({"best_order_id": None, "match_score": round(best_score, 4)})


# ── Direct DB write helpers (not CrewAI tools) ────────────────


def create_shipping_guide(data: dict[str, Any]) -> dict[str, Any] | None:
    """Insert a new shipping guide record. Returns the created record."""
    sb = get_supabase()
    try:
        resp = sb.table("shipping_guides").insert(data).execute()
        if resp.data:
            return resp.data[0]
        return None
    except Exception as exc:
        logger.error("create_shipping_guide error: %s", exc, exc_info=True)
        raise


def send_orphan_notification(
    guide_id: str,
    tracking_code: str,
    carrier_name: str | None,
    recipient_name: str | None,
    recipient_address: str | None,
    recipient_city: str | None,
) -> None:
    """Notify all 'despachos' role users about an orphan guide."""
    sb = get_supabase()
    try:
        # Get dispatch users
        resp = (
            sb.table("user_roles")
            .select("user_id")
            .eq("role", "despachos")
            .execute()
        )
        if not resp.data:
            logger.warning("No dispatch users found for orphan notification")
            return

        notifications = [
            {
                "user_id": u["user_id"],
                "type": "orphan_shipping_guide",
                "title": "Guía sin pedido asociado",
                "message": (
                    f"Se extrajo guía #{tracking_code} de "
                    f"{carrier_name or 'transportadora desconocida'}. "
                    f"Requiere vinculación manual."
                ),
                "severity": "warning",
                "metadata": {
                    "guide_id": guide_id,
                    "tracking_code": tracking_code,
                    "carrier_name": carrier_name,
                    "recipient_name": recipient_name,
                    "recipient_address": recipient_address,
                    "recipient_city": recipient_city,
                },
                "link_to": "/despachos",
            }
            for u in resp.data
        ]

        sb.table("notifications").insert(notifications).execute()
        logger.info(
            "Notified %d dispatch users about orphan guide %s",
            len(resp.data), guide_id,
        )

    except Exception as exc:
        logger.error("send_orphan_notification error: %s", exc, exc_info=True)


def call_order_notify(
    order_id: str,
    carrier_name: str | None,
    tracking_code: str | None,
    tracking_url: str | None,
) -> None:
    """Call the order-notify Edge Function for a matched guide."""
    sb = get_supabase()
    try:
        sb.call_function("order-notify", {
            "order_id": order_id,
            "event": "shipping_guide_linked",
            "carrier": carrier_name,
            "tracking_code": tracking_code,
            "tracking_url": tracking_url,
        })
        logger.info("order-notify called for order %s", order_id)
    except Exception as exc:
        logger.error("call_order_notify error (non-fatal): %s", exc)

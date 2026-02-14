"""
Fuzzy product search against the products table.
Searches name, description, brand, and ai_tags JSONB.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.tools.supabase_client import get_supabase

logger = logging.getLogger(__name__)


def search_products(
    description: str | None = None,
    brand: str | None = None,
    material: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search the products table with fuzzy matching.
    Returns up to 3 best matches.
    """
    sb = get_supabase()

    try:
        query = sb.table("products").select("*")

        # ── Category (exact if provided) ──
        if category:
            query = query.eq("category", category)

        # ── Brand (ilike if provided) ──
        if brand:
            query = query.ilike("brand", f"%{brand}%")

        results = query.execute()
        rows: list[dict[str, Any]] = results.data or []

        if not rows:
            logger.info("products: no rows for brand=%s category=%s", brand, category)
            return []

        # ── Score-based ranking using description match ──
        if description:
            keywords = [w.lower() for w in description.split() if len(w) > 2]
            scored: list[tuple[int, dict[str, Any]]] = []
            for row in rows:
                searchable = " ".join([
                    (row.get("name") or ""),
                    (row.get("description") or ""),
                    (row.get("brand") or ""),
                    (row.get("material") or ""),
                    _flatten_ai_tags(row.get("ai_tags")),
                ]).lower()

                score = sum(1 for kw in keywords if kw in searchable)
                if score > 0:
                    scored.append((score, row))

            scored.sort(key=lambda t: t[0], reverse=True)
            rows = [row for _, row in scored[:3]]

            if not rows:
                logger.info("products: no description match for '%s'", description)
                return []
        else:
            rows = rows[:3]

        # ── Post-filter: material ──
        if material and rows:
            mat_lower = material.strip().lower()
            filtered = [
                r for r in rows
                if mat_lower in (r.get("material") or "").lower()
                or mat_lower in _flatten_ai_tags(r.get("ai_tags")).lower()
            ]
            if filtered:
                rows = filtered

        logger.info(
            "products: returning %d matches for desc=%s brand=%s",
            len(rows), description, brand,
        )
        return rows[:3]

    except Exception as exc:
        logger.error("products search failed: %s", exc, exc_info=True)
        return []


def _flatten_ai_tags(tags: Any) -> str:
    """Convert ai_tags JSONB (dict or None) to a searchable string."""
    if not tags or not isinstance(tags, dict):
        return ""
    return " ".join(str(v) for v in tags.values())

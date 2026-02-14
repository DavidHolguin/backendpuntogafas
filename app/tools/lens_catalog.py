"""
Fuzzy search against the lens_catalog table.
Handles material/treatment normalization for Colombian optical industry terminology.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.tools.supabase_client import get_supabase

logger = logging.getLogger(__name__)

# ── Normalization maps ────────────────────────────────────────
# All keys MUST be lowercase.

MATERIAL_SYNONYMS: dict[str, list[str]] = {
    "policarbonato": ["poly", "poli", "policarbonato", "airwear"],
    "cr": ["cr", "cr-39", "cr39", "resina"],
    "trivex": ["trivex"],
    "cristal": ["cristal", "glass", "vidrio"],
    "hi-index": ["hi-index", "alto indice", "1.67", "1.74"],
}

TREATMENT_SYNONYMS: dict[str, list[str]] = {
    "transitions": ["transitions", "fotocromático", "fotocromatico", "fotosensible", "photochromic"],
    "blue block": ["blue block", "blue", "blue/verde", "blue uv", "blue cut", "blue light"],
    "crizal": ["crizal", "crizal easy", "crizal easy pro", "crizal sapphire", "crizal prevencia"],
    "antireflejo": ["antireflejo", "ar", "anti reflejo", "anti-reflejo"],
    "verde": ["verde", "green"],
}


def _normalize_material(hint: str | None) -> str | None:
    """Map a user material hint to a canonical group key, or None."""
    if not hint:
        return None
    hint_lower = hint.strip().lower()
    for canonical, synonyms in MATERIAL_SYNONYMS.items():
        if hint_lower in synonyms:
            return canonical
    return hint_lower  # fallback: use as-is for ilike


def _normalize_treatment(hint: str | None) -> str | None:
    """Map a user treatment hint to a canonical group key, or None."""
    if not hint:
        return None
    hint_lower = hint.strip().lower()
    for canonical, synonyms in TREATMENT_SYNONYMS.items():
        if hint_lower in synonyms:
            return canonical
        # Also do substring matching
        for syn in synonyms:
            if syn in hint_lower or hint_lower in syn:
                return canonical
    return hint_lower


def _material_patterns(canonical: str | None) -> list[str]:
    """Expand a canonical material to all DB-friendly ILIKE patterns."""
    if not canonical:
        return []
    synonyms = MATERIAL_SYNONYMS.get(canonical, [canonical])
    return [f"%{s}%" for s in synonyms]


def _treatment_patterns(canonical: str | None) -> list[str]:
    """Expand a canonical treatment to all DB-friendly ILIKE patterns."""
    if not canonical:
        return []
    synonyms = TREATMENT_SYNONYMS.get(canonical, [canonical])
    return [f"%{s}%" for s in synonyms]


def search_lens_catalog(
    category: str | None = None,
    material_hint: str | None = None,
    treatment_hint: str | None = None,
    is_digital: bool | None = None,
    sphere: float | None = None,
    cylinder: float | None = None,
    add_power: float | None = None,
) -> list[dict[str, Any]]:
    """
    Search lens_catalog with fuzzy material/treatment matching.
    Returns up to 3 best matches sorted by price (cheapest first).
    """
    sb = get_supabase()

    try:
        query = sb.table("lens_catalog").select("*").eq("active", True)

        # ── Category (exact) ──
        if category:
            query = query.eq("category", category.lower())

        # ── Digital ──
        if is_digital is not None:
            query = query.eq("is_digital", is_digital)

        results = query.execute()
        rows: list[dict[str, Any]] = results.data or []

        if not rows:
            logger.info("lens_catalog: no active rows for category=%s", category)
            return []

        # ── Post-filter: material (fuzzy) ──
        canon_material = _normalize_material(material_hint)
        if canon_material:
            patterns = _material_patterns(canon_material)
            filtered = []
            for row in rows:
                row_mat = (row.get("material") or "").lower()
                row_type = (row.get("lens_type") or "").lower()
                combined = f"{row_mat} {row_type}"
                if any(p.strip("%") in combined for p in patterns):
                    filtered.append(row)
            if filtered:
                rows = filtered
            # If no match, keep all rows (better partial match than nothing)

        # ── Post-filter: treatment (fuzzy) ──
        canon_treatment = _normalize_treatment(treatment_hint)
        if canon_treatment:
            patterns = _treatment_patterns(canon_treatment)
            filtered = []
            for row in rows:
                row_treat = (row.get("treatment") or "").lower()
                row_type = (row.get("lens_type") or "").lower()
                combined = f"{row_treat} {row_type}"
                if any(p.strip("%") in combined for p in patterns):
                    filtered.append(row)
            if filtered:
                rows = filtered

        # ── Post-filter: optical ranges ──
        if sphere is not None:
            rows = [
                r for r in rows
                if (r.get("sphere_min") is None or float(r["sphere_min"]) <= sphere)
                and (r.get("sphere_max") is None or float(r["sphere_max"]) >= sphere)
            ]
        if cylinder is not None:
            rows = [
                r for r in rows
                if (r.get("cylinder_min") is None or float(r["cylinder_min"]) <= cylinder)
                and (r.get("cylinder_max") is None or float(r["cylinder_max"]) >= cylinder)
            ]
        if add_power is not None:
            rows = [
                r for r in rows
                if (r.get("add_min") is None or float(r["add_min"]) <= add_power)
                and (r.get("add_max") is None or float(r["add_max"]) >= add_power)
            ]

        # ── Sort by retail_price ascending, return top 3 ──
        rows.sort(key=lambda r: float(r.get("retail_price", 0) or 0))
        top = rows[:3]

        logger.info(
            "lens_catalog: found %d matches (returning %d) for cat=%s mat=%s treat=%s",
            len(rows), len(top), category, material_hint, treatment_hint,
        )
        return top

    except Exception as exc:
        logger.error("lens_catalog search failed: %s", exc, exc_info=True)
        return []

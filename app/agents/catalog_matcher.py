"""
Agent 3: Catalog Matcher — cross-references extracted items with
lens_catalog and products tables in Supabase.
No LLM call needed; this is pure search + matching logic.

Sale-tag aware:
  - For venta_directa, only search products (skip lens catalog entirely)
  - For estuche items, search products with category hints for accessories
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.extraction import (
    AlternativeMatch,
    CatalogOutput,
    ConversationOutput,
    ItemRequested,
    MatchedItem,
    VisionOutput,
)
from app.tools.lens_catalog import search_lens_catalog
from app.tools.products import search_products

logger = logging.getLogger(__name__)


def _get_rx_values(vision: VisionOutput) -> dict[str, float | None]:
    """Extract representative sphere/cylinder/add from the first prescription."""
    if not vision.prescriptions_found:
        return {"sphere": None, "cylinder": None, "add_power": None}

    rx = vision.prescriptions_found[0].rx_data
    if not rx:
        return {"sphere": None, "cylinder": None, "add_power": None}

    # Use the "worse" eye (higher absolute sphere) for range matching
    sphere = None
    cylinder = None
    add_power = None

    if rx.od and rx.od.sphere is not None:
        sphere = rx.od.sphere
    if rx.os and rx.os.sphere is not None:
        if sphere is None or abs(rx.os.sphere) > abs(sphere):
            sphere = rx.os.sphere

    if rx.od and rx.od.cylinder is not None:
        cylinder = rx.od.cylinder
    if rx.os and rx.os.cylinder is not None:
        if cylinder is None or abs(rx.os.cylinder) > abs(cylinder):
            cylinder = rx.os.cylinder

    if rx.od and rx.od.add is not None:
        add_power = rx.od.add
    elif rx.os and rx.os.add is not None:
        add_power = rx.os.add

    return {"sphere": sphere, "cylinder": cylinder, "add_power": add_power}


def _match_lens(
    item: ItemRequested,
    rx_values: dict[str, float | None],
) -> MatchedItem:
    """Match a lens item against lens_catalog."""
    results = search_lens_catalog(
        category=item.category,
        material_hint=item.material_hint,
        treatment_hint=item.treatment_hint,
        is_digital=item.is_digital,
        sphere=rx_values.get("sphere"),
        cylinder=rx_values.get("cylinder"),
        add_power=rx_values.get("add_power"),
    )

    if not results:
        logger.info("No lens catalog match for: %s", item.description)
        return MatchedItem(
            type="lente",
            description=item.description or "Lente - sin match en catálogo",
            unit_price=0,
            quantity=item.quantity,
            confidence=0.0,
            needs_manual_selection=True,
            notes=item.notes,
        )

    best = results[0]
    alternatives = [
        AlternativeMatch(
            lens_catalog_id=r["id"],
            description=r.get("lens_type", ""),
            price=float(r.get("retail_price", 0) or 0),
        )
        for r in results[1:3]
    ]

    confidence = 0.9 if len(results) >= 1 else 0.5

    return MatchedItem(
        type="lente",
        lens_catalog_id=best["id"],
        lab_id=best.get("lab_id"),
        description=best.get("lens_type", item.description),
        unit_price=float(best.get("retail_price", 0) or 0),
        lab_cost=float(best.get("lab_cost", 0) or 0),
        quantity=item.quantity,
        confidence=confidence,
        alternatives=alternatives,
        notes=item.notes,
    )


def _match_product(item: ItemRequested) -> MatchedItem:
    """Match a frame/accessory item against products table."""
    results = search_products(
        description=item.description,
        brand=item.brand_hint,
        material=item.material_hint,
        category=item.type,  # "montura", "accesorio"
    )

    if not results:
        logger.info("No product match for: %s", item.description)
        desc = item.description or f"{item.type or 'Producto'} - Pendiente selección"
        return MatchedItem(
            type=item.type or "montura",
            description=desc,
            unit_price=0,
            quantity=item.quantity,
            confidence=0.0,
            needs_manual_selection=True,
            notes=item.notes,
        )

    best = results[0]
    alternatives = [
        AlternativeMatch(
            product_id=r["id"],
            description=r.get("name", ""),
            price=float(r.get("price", 0) or 0),
        )
        for r in results[1:3]
    ]

    return MatchedItem(
        type=item.type or "montura",
        product_id=best["id"],
        description=best.get("name", item.description),
        unit_price=float(best.get("price", 0) or 0),
        quantity=item.quantity,
        confidence=0.8,
        alternatives=alternatives,
        notes=item.notes,
    )


# ── Public API ────────────────────────────────────────────────

def run_catalog_matcher(
    conversation: ConversationOutput,
    vision: VisionOutput,
) -> CatalogOutput:
    """
    Match extracted items against catalog and product databases.

    For venta_directa orders:
      - Only search products table (monturas/accesorios)
      - Skip lens catalog entirely
      - No lab_id suggestion

    Always returns a valid CatalogOutput, even if no matches found.
    """
    is_venta_directa = conversation.suggested_order_type == "venta_directa"

    if not conversation.items_requested:
        logger.info("Catalog matcher: no items to match")
        return CatalogOutput(
            warnings=["No hay items para buscar en catálogo"],
        )

    rx_values = _get_rx_values(vision) if not is_venta_directa else {}
    matched: list[MatchedItem] = []
    warnings: list[str] = []
    suggested_lab_id: str | None = None

    for item in conversation.items_requested:
        try:
            if is_venta_directa:
                # ── Venta directa: only products, no lens catalog ──
                match = _match_product(item)
                logger.info(
                    "Venta directa match: %s → %s ($%.0f)",
                    item.type, match.description, match.unit_price,
                )
            elif item.type == "lente":
                match = _match_lens(item, rx_values)
                # Track suggested lab from the first matched lens
                if match.lab_id and not suggested_lab_id:
                    suggested_lab_id = match.lab_id
            elif item.type in ("montura", "accesorio"):
                match = _match_product(item)
            elif item.type == "servicio":
                match = MatchedItem(
                    type="servicio",
                    description=item.description or "Servicio",
                    unit_price=0,
                    quantity=item.quantity,
                    needs_manual_selection=True,
                    notes=item.notes,
                )
            else:
                # Unknown type — create as manual selection
                match = MatchedItem(
                    type=item.type or "otro",
                    description=item.description or "Item no clasificado",
                    unit_price=0,
                    quantity=item.quantity,
                    needs_manual_selection=True,
                    notes=item.notes,
                )

            if match.needs_manual_selection:
                warnings.append(
                    f"Sin match para {item.type}: '{item.description}' — logística debe asignar"
                )

            matched.append(match)

        except Exception as exc:
            logger.error("Error matching item '%s': %s", item.description, exc, exc_info=True)
            # Still add the item with zero price
            matched.append(MatchedItem(
                type=item.type or "otro",
                description=item.description or "Item con error de matching",
                unit_price=0,
                quantity=item.quantity,
                needs_manual_selection=True,
                notes=f"Error: {exc}",
            ))
            warnings.append(f"Error al buscar '{item.description}': {exc}")

    logger.info(
        "Catalog matcher: %d items matched, %d warnings, lab=%s, venta_directa=%s",
        len(matched), len(warnings), suggested_lab_id, is_venta_directa,
    )

    return CatalogOutput(
        matched_items=matched,
        warnings=warnings,
        suggested_lab_id=suggested_lab_id if not is_venta_directa else None,
    )

"""
Pydantic models for intermediate outputs of the extraction pipeline.
Covers: VisionOutput, ConversationOutput, CatalogOutput.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .prescription import NonPrescriptionImage, PrescriptionFound


# ── Agent 1: Vision Extractor output ──────────────────────────

class VisionOutput(BaseModel):
    prescriptions_found: list[PrescriptionFound] = Field(default_factory=list)
    non_prescription_images: list[NonPrescriptionImage] = Field(default_factory=list)
    error: Optional[str] = None


# ── Agent 2: Conversation Analyzer output ─────────────────────

class ItemRequested(BaseModel):
    type: Optional[str] = None          # "lente", "montura", "accesorio", "servicio"
    description: Optional[str] = None
    category: Optional[str] = None      # "progresivo", "monofocal", "bifocal", "ocupacional"
    material_hint: Optional[str] = None
    treatment_hint: Optional[str] = None
    is_digital: Optional[bool] = None
    brand_hint: Optional[str] = None
    model_hint: Optional[str] = None
    quantity: int = 1
    notes: Optional[str] = None


class CustomerUpdates(BaseModel):
    """Data mentioned in the conversation that could update the customer record."""
    email: Optional[str] = None
    document_id: Optional[str] = None
    city: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None


class ConversationOutput(BaseModel):
    items_requested: list[ItemRequested] = Field(default_factory=list)
    special_instructions: Optional[str] = None
    urgency: str = "desconocida"
    promised_date_hint: Optional[str] = None
    customer_updates: Optional[CustomerUpdates] = None
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


# ── Agent 3: Catalog Matcher output ───────────────────────────

class AlternativeMatch(BaseModel):
    lens_catalog_id: Optional[str] = None
    product_id: Optional[str] = None
    description: Optional[str] = None
    price: float = 0


class MatchedItem(BaseModel):
    type: Optional[str] = None           # "lente", "montura", "accesorio"
    lens_catalog_id: Optional[str] = None
    lab_id: Optional[str] = None
    product_id: Optional[str] = None
    description: Optional[str] = None
    unit_price: float = 0
    lab_cost: Optional[float] = None
    quantity: int = 1
    confidence: float = 0.0
    needs_manual_selection: bool = False
    alternatives: list[AlternativeMatch] = Field(default_factory=list)
    notes: Optional[str] = None


class CatalogOutput(BaseModel):
    matched_items: list[MatchedItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggested_lab_id: Optional[str] = None
    error: Optional[str] = None

"""
Pydantic models for the final order draft result —
the output that gets stored in ai_order_jobs.result and written to DB.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .extraction import CustomerUpdates
from .prescription import AiExtractionMetadata, RxData


# ── Order header for INSERT into orders ───────────────────────

class OrderDraftHeader(BaseModel):
    customer_id: Optional[str] = None
    sede_id: Optional[str] = None
    seller_id: Optional[str] = None
    status: str = "borrador"
    total_amount: float = 0
    balance_due: float = 0
    payment_status: str = "pendiente"
    lab_id: Optional[str] = None
    promised_date: Optional[str] = None


# ── Individual order item ─────────────────────────────────────

class OrderDraftItem(BaseModel):
    description: Optional[str] = None
    quantity: int = 1
    unit_price: float = 0
    subtotal: float = 0
    lens_catalog_id: Optional[str] = None
    lens_lab_cost: Optional[float] = None
    product_id: Optional[str] = None
    prescription_id: Optional[str] = None  # Filled after prescription INSERT
    needs_manual_selection: bool = False


# ── Prescription for INSERT ───────────────────────────────────

class PrescriptionInsert(BaseModel):
    customer_id: Optional[str] = None
    rx_data: Optional[RxData] = None
    original_image_url: Optional[str] = None
    ai_extraction_metadata: Optional[AiExtractionMetadata] = None


# ── Complete pipeline result ──────────────────────────────────

class FinalOrderResult(BaseModel):
    order_draft: OrderDraftHeader = Field(default_factory=OrderDraftHeader)
    items: list[OrderDraftItem] = Field(default_factory=list)
    prescription: Optional[PrescriptionInsert] = None
    customer_updates: Optional[CustomerUpdates] = None
    completeness: str = "minimo"          # "completo", "parcial", "minimo"
    warnings: list[str] = Field(default_factory=list)
    needs_manual_review: bool = True
    processing_time_ms: int = 0
    agent_errors: dict[str, str] = Field(default_factory=dict)  # agent_name -> error

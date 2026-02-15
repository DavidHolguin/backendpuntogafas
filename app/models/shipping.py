"""
Pydantic models for the Shipping Guide Extraction API.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────


class ShippingExtractRequest(BaseModel):
    """Payload sent by whatsapp-media-processor."""

    image_url: str = Field(..., description="Public URL of the shipping guide image")
    message_id: Optional[str] = Field(None, description="UUID of the WhatsApp message")
    customer_id: Optional[str] = Field(None, description="UUID of the transportadora customer")
    conversation_id: Optional[str] = Field(None, description="UUID of the conversation")
    carrier_company_id: Optional[str] = Field(None, description="UUID from carrier_contacts lookup")


# ── Extracted Data ────────────────────────────────────────────


class ExtractedGuideData(BaseModel):
    """Structured data extracted from a shipping guide image."""

    tracking_code: Optional[str] = None
    carrier_name: Optional[str] = None
    tracking_url: Optional[str] = None
    recipient_name: Optional[str] = None
    recipient_address: Optional[str] = None
    recipient_city: Optional[str] = None
    confidence: float = 0.0


# ── Response ──────────────────────────────────────────────────


class ShippingExtractResponse(BaseModel):
    """Response from the shipping guide extraction endpoint."""

    success: bool
    guide_id: Optional[str] = None
    order_id: Optional[str] = None
    is_orphan: bool = False
    match_score: float = 0.0
    extracted: Optional[ExtractedGuideData] = None
    error: Optional[str] = None
    duplicate: bool = False

"""
Pydantic models for optical prescriptions (rx_data).
All numeric fields are Optional — partial data is expected.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EyeRx(BaseModel):
    """Single eye prescription values."""
    sphere: Optional[float] = None
    cylinder: Optional[float] = None
    axis: Optional[int] = None
    add: Optional[float] = None


class PupilDistance(BaseModel):
    right: Optional[float] = None
    left: Optional[float] = None


class RxData(BaseModel):
    """The rx_data JSONB structure stored in prescriptions table."""
    od: Optional[EyeRx] = None          # Ojo derecho
    os: Optional[EyeRx] = None          # Ojo izquierdo
    pd: Optional[PupilDistance] = None   # Distancia pupilar
    notes: Optional[str] = None


class PrescriptionFound(BaseModel):
    """A prescription extracted from an image by the Vision agent."""
    image_url: Optional[str] = None
    rx_data: Optional[RxData] = None
    confidence: float = 0.0
    warnings: list[str] = []
    notes: Optional[str] = None


class NonPrescriptionImage(BaseModel):
    """An image that is NOT a prescription (e.g. frame photo)."""
    image_url: Optional[str] = None
    description: Optional[str] = None


class AiExtractionMetadata(BaseModel):
    source: str = "ai_extracted"
    confidence: float = 0.0
    model: str = "gemini-2.0-flash"
    warnings: list[str] = []
    extracted_at: Optional[str] = None

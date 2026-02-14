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


# ── Remission data (extracted from Punto Gafas remission docs) ──

class WarrantyInfo(BaseModel):
    frame: Optional[str] = None          # ej: "1 año"
    lens: Optional[str] = None           # ej: "6 meses Blue"
    conditions: list[str] = []           # ej: ["no golpes", "no rayones"]


class RemissionData(BaseModel):
    """Data extracted from a Punto Gafas remission document."""
    image_url: Optional[str] = None
    lens_description: Optional[str] = None      # ej: "Blue Block Poli"
    warranty: Optional[WarrantyInfo] = None
    delivery_days: Optional[int] = None          # ej: 12
    payment_info: Optional[str] = None           # ej: "Pago completo - Datáfono"
    observations: Optional[str] = None           # ej: "URGENTE"
    total_amount: Optional[float] = None         # ej: 160000
    remission_number: Optional[str] = None       # ej: "10241"
    confidence: float = 0.0


# ── Clinical history data ─────────────────────────────────────

class VisualAcuity(BaseModel):
    vp_od: Optional[str] = None   # Visión próxima OD
    vp_os: Optional[str] = None   # Visión próxima OS
    vl_od: Optional[str] = None   # Visión lejana OD (ej: "20/20")
    vl_os: Optional[str] = None   # Visión lejana OS


class ClinicalHistoryData(BaseModel):
    """Data extracted from the clinical history section of a prescription."""
    image_url: Optional[str] = None
    diagnosis_od: Optional[str] = None
    diagnosis_os: Optional[str] = None
    visual_acuity: Optional[VisualAcuity] = None
    next_control: Optional[str] = None
    professional_name: Optional[str] = None
    confidence: float = 0.0


# ── Image classification ──────────────────────────────────────

class ImageClassification(BaseModel):
    """Classification of each image processed by the Vision agent."""
    url: Optional[str] = None
    type: str = "other"   # "formula", "remission", "frame", "clinical_history", "other"
    confidence: float = 0.0

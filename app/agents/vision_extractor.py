"""
Agent 1: Vision Extractor v3 — Classifies images into 4 types
and extracts structured data per type, including payment info.

Types:
  1. FORMULA ÓPTICA → rx_data (OD/OS/PD) + optional embedded clinical_history
  2. REMISIÓN → lens, warranty, delivery, payment method/type/amount
  3. MONTURA → frame reference code, description
  4. HISTORIAL CLÍNICO → diagnosis, visual acuity
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from google import genai
from google.genai import types as genai_types

from app.config import settings
from app.models.extraction import VisionOutput
from app.models.prescription import (
    ClinicalHistoryData,
    EyeRx,
    FrameData,
    ImageClassification,
    NonPrescriptionImage,
    PrescriptionFound,
    PupilDistance,
    RemissionData,
    RxData,
    VisualAcuity,
    WarrantyInfo,
)

logger = logging.getLogger(__name__)

# ── Gemini V3 prompt ──────────────────────────────────────────

VISION_V3_PROMPT = """Eres un extractor de datos ópticos especializado para Punto Gafas Colombia.
Recibes imágenes de una conversación de WhatsApp entre un asesor y un cliente.

CLASIFICACIÓN DE IMÁGENES (4 tipos):
1. FORMULA ÓPTICA: Certificado con valores refractivos OD/OS
2. REMISIÓN: Documento de remisión de Punto Gafas
3. MONTURA: Foto de la montura/armazón seleccionada
4. HISTORIAL CLÍNICO: Diagnóstico y agudeza visual

PASO 1 - CLASIFICAR la imagen en uno de los 4 tipos.
PASO 2 - EXTRAER datos según el tipo y responder SOLO con JSON:

Para FORMULA ÓPTICA:
{
  "image_type": "formula",
  "confidence": <0.0-1.0>,
  "rx_data": {
    "od": {"sphere": <número o null>, "cylinder": <número o null>, "axis": <entero o null>, "add": <número o null>},
    "os": {"sphere": <número o null>, "cylinder": <número o null>, "axis": <entero o null>, "add": <número o null>},
    "pd": {"right": <número o null>, "left": <número o null>}
  },
  "patient_name": <string o null>,
  "document_id": <string o null>,
  "warnings": ["lista de advertencias"],
  "notes": "observaciones",
  "clinical_history": null o {
    "diagnosis_od": <string o null>,
    "diagnosis_os": <string o null>,
    "av_vp_od": <string o null>,
    "av_vp_os": <string o null>,
    "av_vl_od": <string o null>,
    "av_vl_os": <string o null>,
    "next_control": <string o null>,
    "professional_name": <string o null>,
    "confidence": <0.0-1.0>
  }
}

Para REMISIÓN:
{
  "image_type": "remission",
  "confidence": <0.0-1.0>,
  "lens_description": <string exacta del lente ej: "Blue Block Poli">,
  "warranty_frame": <string ej: "1 año">,
  "warranty_lens": <string ej: "6 meses Blue">,
  "warranty_conditions": [<lista de condiciones ej: "no golpes">],
  "delivery_days": <número entero o null>,
  "observations": <string o null>,
  "remission_number": <string o null>,
  "total_amount": <número o null>,
  "payment_method": <string mapeado: ver reglas abajo>,
  "payment_type": "total" o "parcial",
  "payment_amount": <número o null>
}

MAPEO DE MÉTODOS DE PAGO:
- "Datáfono" o "Datafono" = "tarjeta"
- "Nequi" = "nequi"
- "Daviplata" = "daviplata"
- "Efectivo" = "efectivo"
- "Transferencia" o "Consignación" = "transferencia"

Para HISTORIAL CLÍNICO:
{
  "image_type": "clinical_history",
  "confidence": <0.0-1.0>,
  "diagnosis_od": <string o null>,
  "diagnosis_os": <string o null>,
  "av_vp_od": <string o null>,
  "av_vp_os": <string o null>,
  "av_vl_od": <string o null>,
  "av_vl_os": <string o null>,
  "next_control": <string o null>,
  "professional_name": <string o null>
}

Para MONTURA:
{
  "image_type": "frame",
  "confidence": <0.0-1.0>,
  "description": "descripción breve de la montura",
  "reference_code": <string o null si es visible>
}

REGLAS:
- Si no puedes leer un valor, usar null, NUNCA inventar datos
- Los valores de sphere y cylinder pueden ser positivos o negativos (ej: -2.00, +1.50)
- El axis es un entero entre 0 y 180
- La adición (add) es siempre positiva (ej: 1.50, 2.00)
- PD puede ser un solo valor total o separado derecho/izquierdo
- OD = Ojo Derecho, OS = Ojo Izquierdo
- DNP, DIP, DP son sinónimos de PD
- Una misma imagen puede contener fórmula + historial clínico (parte superior e inferior). Si es así, incluye clinical_history dentro de la respuesta de fórmula.
- NUNCA usar el monto de la remisión como precio final del pedido
- El monto de pago es REFERENCIAL, el sistema calcula el total desde el catálogo de lentes
- Reportar confidence 0.0-1.0 por imagen
- Responde SOLO con el JSON, sin texto adicional"""


def _download_image(url: str) -> bytes | None:
    """Download an image from a URL. Returns bytes or None on failure."""
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        logger.error("Failed to download image %s: %s", url, exc)
        return None


def _call_gemini_vision(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict[str, Any]:
    """Send an image to Gemini and get structured extraction."""
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[
                    genai_types.Content(
                        parts=[
                            genai_types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type,
                            ),
                            genai_types.Part.from_text(text=VISION_V3_PROMPT),
                        ]
                    )
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=3000,
                ),
            )

            text = response.text or ""
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            return json.loads(text)

        except json.JSONDecodeError as exc:
            logger.warning("Gemini returned invalid JSON (attempt %d): %s", attempt + 1, exc)
            if attempt == 2:
                return {"image_type": "other", "description": "Error: respuesta no parseable de IA"}

        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                delay = settings.RETRY_BASE_DELAY * (2 ** attempt)
                delay = min(delay, settings.RETRY_MAX_DELAY)
                logger.warning("Gemini rate limit (attempt %d), waiting %.1fs", attempt + 1, delay)
                time.sleep(delay)
            else:
                logger.error("Gemini vision error (attempt %d): %s", attempt + 1, exc)
                if attempt == 2:
                    return {"image_type": "other", "description": f"Error de IA: {exc}"}

    return {"image_type": "other", "description": "Error: agotados reintentos de IA"}


def _guess_mime_type(url: str) -> str:
    url_lower = url.lower().split("?")[0]
    if url_lower.endswith(".png"):
        return "image/png"
    if url_lower.endswith(".webp"):
        return "image/webp"
    if url_lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


# ── Parsers by image type ─────────────────────────────────────

def _parse_formula(data: dict[str, Any], url: str) -> tuple[
    PrescriptionFound, ClinicalHistoryData | None
]:
    """Parse a formula image response into PrescriptionFound + optional ClinicalHistory."""
    rx_raw = data.get("rx_data", {})
    od_raw = rx_raw.get("od", {})
    os_raw = rx_raw.get("os", {})
    pd_raw = rx_raw.get("pd", {})

    rx_data = RxData(
        od=EyeRx(**od_raw) if od_raw else None,
        os=EyeRx(**os_raw) if os_raw else None,
        pd=PupilDistance(**pd_raw) if pd_raw else None,
        notes=data.get("notes"),
    )

    prescription = PrescriptionFound(
        image_url=url,
        rx_data=rx_data,
        confidence=float(data.get("confidence", 0.5)),
        warnings=data.get("warnings", []),
        notes=data.get("notes"),
    )

    # Check if clinical history is embedded in the same image
    clinical = None
    ch_raw = data.get("clinical_history")
    if ch_raw and isinstance(ch_raw, dict):
        clinical = _parse_clinical_raw(ch_raw, url)

    return prescription, clinical


def _parse_remission(data: dict[str, Any], url: str) -> RemissionData:
    """Parse a remission document response with structured payment data."""
    warranty = None
    if data.get("warranty_frame") or data.get("warranty_lens") or data.get("warranty_conditions"):
        warranty = WarrantyInfo(
            frame=data.get("warranty_frame"),
            lens=data.get("warranty_lens"),
            conditions=data.get("warranty_conditions", []),
        )

    # Build raw payment_info text for backward compat
    payment_parts = []
    if data.get("payment_type"):
        payment_parts.append("Pago " + data["payment_type"])
    if data.get("payment_method"):
        payment_parts.append(data["payment_method"])
    payment_info = " - ".join(payment_parts) if payment_parts else None

    return RemissionData(
        image_url=url,
        lens_description=data.get("lens_description"),
        warranty=warranty,
        delivery_days=int(data["delivery_days"]) if data.get("delivery_days") else None,
        payment_info=payment_info,
        payment_method=data.get("payment_method"),
        payment_type=data.get("payment_type"),
        payment_amount=float(data["payment_amount"]) if data.get("payment_amount") else None,
        has_proof=bool(data.get("has_proof", False)),
        observations=data.get("observations"),
        total_amount=float(data["total_amount"]) if data.get("total_amount") else None,
        remission_number=data.get("remission_number"),
        confidence=float(data.get("confidence", 0.5)),
    )


def _parse_clinical_raw(data: dict[str, Any], url: str) -> ClinicalHistoryData:
    """Parse raw clinical history data (standalone or embedded in formula)."""
    va = None
    if any(data.get(k) for k in ("av_vp_od", "av_vp_os", "av_vl_od", "av_vl_os")):
        va = VisualAcuity(
            vp_od=data.get("av_vp_od"),
            vp_os=data.get("av_vp_os"),
            vl_od=data.get("av_vl_od"),
            vl_os=data.get("av_vl_os"),
        )

    return ClinicalHistoryData(
        image_url=url,
        diagnosis_od=data.get("diagnosis_od"),
        diagnosis_os=data.get("diagnosis_os"),
        visual_acuity=va,
        next_control=data.get("next_control"),
        professional_name=data.get("professional_name"),
        confidence=float(data.get("confidence", 0.5)),
    )


def _parse_frame(data: dict[str, Any], url: str) -> FrameData:
    """Parse a frame/montura image response."""
    return FrameData(
        image_url=url,
        reference_code=data.get("reference_code"),
        description=data.get("description", "Montura"),
        confidence=float(data.get("confidence", 0.5)),
    )


# ── Public API ────────────────────────────────────────────────

def run_vision_extractor(media_urls: list[str]) -> VisionOutput:
    """
    Process all media_urls through Gemini Vision v3.
    Classifies each image and extracts structured data per type.
    Always returns a valid VisionOutput, even on complete failure.
    """
    if not media_urls:
        logger.info("Vision extractor: no media_urls provided, skipping")
        return VisionOutput()

    prescriptions: list[PrescriptionFound] = []
    non_prescriptions: list[NonPrescriptionImage] = []
    remissions: list[RemissionData] = []
    clinical_histories: list[ClinicalHistoryData] = []
    frames: list[FrameData] = []
    classifications: list[ImageClassification] = []

    for url in media_urls:
        logger.info("Processing image: %s", url)

        image_bytes = _download_image(url)
        if not image_bytes:
            non_prescriptions.append(NonPrescriptionImage(
                image_url=url,
                description="Error: no se pudo descargar la imagen",
            ))
            classifications.append(ImageClassification(url=url, type="other", confidence=0.0))
            continue

        mime_type = _guess_mime_type(url)
        result = _call_gemini_vision(image_bytes, mime_type)
        image_type = result.get("image_type", "other")
        confidence = float(result.get("confidence", 0.5))

        # Always add classification
        classifications.append(ImageClassification(
            url=url,
            type=image_type,
            confidence=confidence,
        ))

        try:
            if image_type == "formula":
                prescription, clinical = _parse_formula(result, url)
                prescriptions.append(prescription)
                logger.info("Formula extracted (confidence: %.2f)", prescription.confidence)

                if clinical:
                    clinical_histories.append(clinical)
                    logger.info("Clinical history embedded in formula")
                    classifications.append(ImageClassification(
                        url=url, type="clinical_history", confidence=clinical.confidence,
                    ))

            elif image_type == "remission":
                remission = _parse_remission(result, url)
                remissions.append(remission)
                logger.info(
                    "Remission extracted: %s, payment=%s/%s (conf: %.2f)",
                    remission.lens_description,
                    remission.payment_method,
                    remission.payment_type,
                    remission.confidence,
                )

            elif image_type == "clinical_history":
                clinical = _parse_clinical_raw(result, url)
                clinical_histories.append(clinical)
                logger.info("Clinical history extracted (confidence: %.2f)", clinical.confidence)

            elif image_type == "frame":
                frame = _parse_frame(result, url)
                frames.append(frame)
                logger.info("Frame classified: %s (ref: %s)", frame.description, frame.reference_code)

            else:
                non_prescriptions.append(NonPrescriptionImage(
                    image_url=url,
                    description=result.get("description", "Imagen no clasificada"),
                ))
                logger.info("Other image: %s", result.get("description"))

        except Exception as exc:
            logger.error("Error parsing %s result for %s: %s", image_type, url, exc, exc_info=True)
            non_prescriptions.append(NonPrescriptionImage(
                image_url=url,
                description=f"Error al parsear: {exc}",
            ))

    return VisionOutput(
        prescriptions_found=prescriptions,
        non_prescription_images=non_prescriptions,
        remissions=remissions,
        clinical_histories=clinical_histories,
        frames=frames,
        image_classifications=classifications,
    )

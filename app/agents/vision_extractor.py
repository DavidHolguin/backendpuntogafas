"""
Agent 1: Vision Extractor — OCR for optometric prescriptions.
Downloads images from media_urls, sends each to Gemini 2.0 Flash (multimodal),
and extracts rx_data from prescription images.
Non-prescription images (frames, selfies) are classified separately.
"""

from __future__ import annotations

import base64
import io
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
    EyeRx,
    NonPrescriptionImage,
    PrescriptionFound,
    PupilDistance,
    RxData,
)

logger = logging.getLogger(__name__)

# ── Gemini prompt for prescription extraction ─────────────────

PRESCRIPTION_PROMPT = """Eres un experto en optometría analizando una imagen.

TAREA: Determinar si la imagen contiene una fórmula óptica (prescripción optométrica / receta de lentes).

SI ES UNA FÓRMULA ÓPTICA, extrae los siguientes datos en formato JSON:
{
  "is_prescription": true,
  "rx_data": {
    "od": {"sphere": <número o null>, "cylinder": <número o null>, "axis": <entero o null>, "add": <número o null>},
    "os": {"sphere": <número o null>, "cylinder": <número o null>, "axis": <entero o null>, "add": <número o null>},
    "pd": {"right": <número o null>, "left": <número o null>}
  },
  "confidence": <0.0 a 1.0>,
  "warnings": ["lista de advertencias si algo es ilegible o dudoso"],
  "notes": "observaciones adicionales como diagnóstico o recomendaciones"
}

SI NO ES UNA FÓRMULA (foto de montura, selfie, documento, etc):
{
  "is_prescription": false,
  "description": "descripción breve de lo que muestra la imagen"
}

REGLAS:
- Los valores de sphere y cylinder pueden ser positivos o negativos (ej: -2.00, +1.50)
- El axis es un entero entre 0 y 180
- El add (adición) es siempre positivo (ej: 1.50, 2.00)
- PD (distancia pupilar) puede ser un solo valor total o separado derecho/izquierdo
- Si un dato es ilegible, pon null y agrega un warning
- Si la confianza es baja (< 0.5), indica en warnings qué valores son dudosos
- OD = Ojo Derecho (Right Eye), OS = Ojo Izquierdo (Left Eye)
- Busca también: DNP, DIP, DP como sinónimos de PD
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
    """Send an image to Gemini 2.0 Flash and get structured extraction."""
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # Exponential backoff for rate limits
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
                            genai_types.Part.from_text(text=PRESCRIPTION_PROMPT),
                        ]
                    )
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2000,
                ),
            )

            text = response.text or ""
            # Clean markdown code fences if present
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
                return {"is_prescription": False, "description": "Error: respuesta no parseable de IA"}

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
                    return {"is_prescription": False, "description": f"Error de IA: {exc}"}

    return {"is_prescription": False, "description": "Error: agotados reintentos de IA"}


def _guess_mime_type(url: str) -> str:
    """Guess MIME type from URL extension."""
    url_lower = url.lower()
    if url_lower.endswith(".png"):
        return "image/png"
    if url_lower.endswith(".webp"):
        return "image/webp"
    if url_lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def _parse_extraction(data: dict[str, Any], image_url: str) -> PrescriptionFound | NonPrescriptionImage:
    """Convert raw Gemini output to a typed model."""
    if data.get("is_prescription"):
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

        return PrescriptionFound(
            image_url=image_url,
            rx_data=rx_data,
            confidence=float(data.get("confidence", 0.5)),
            warnings=data.get("warnings", []),
            notes=data.get("notes"),
        )
    else:
        return NonPrescriptionImage(
            image_url=image_url,
            description=data.get("description", "Imagen no clasificada"),
        )


# ── Public API ────────────────────────────────────────────────

def run_vision_extractor(media_urls: list[str]) -> VisionOutput:
    """
    Process all media_urls through Gemini Vision.
    Always returns a valid VisionOutput, even on complete failure.
    """
    if not media_urls:
        logger.info("Vision extractor: no media_urls provided, skipping")
        return VisionOutput()

    prescriptions: list[PrescriptionFound] = []
    non_prescriptions: list[NonPrescriptionImage] = []

    for url in media_urls:
        logger.info("Processing image: %s", url)

        image_bytes = _download_image(url)
        if not image_bytes:
            non_prescriptions.append(NonPrescriptionImage(
                image_url=url,
                description="Error: no se pudo descargar la imagen",
            ))
            continue

        mime_type = _guess_mime_type(url)
        result = _call_gemini_vision(image_bytes, mime_type)
        parsed = _parse_extraction(result, url)

        if isinstance(parsed, PrescriptionFound):
            prescriptions.append(parsed)
            logger.info("Prescription extracted (confidence: %.2f)", parsed.confidence)
        else:
            non_prescriptions.append(parsed)
            logger.info("Non-prescription image: %s", parsed.description)

    return VisionOutput(
        prescriptions_found=prescriptions,
        non_prescription_images=non_prescriptions,
    )

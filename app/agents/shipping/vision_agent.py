"""
Vision Agent — Extracts structured data from shipping guide images.
Uses Gemini 2.0 Flash for vision analysis of Colombian shipping labels.
"""

from __future__ import annotations

import json
import logging
import time

from google import genai
from google.genai import types

from app.config import settings
from app.models.shipping import ExtractedGuideData

logger = logging.getLogger(__name__)

# ── System prompt for guide extraction ────────────────────────

EXTRACTION_SYSTEM_PROMPT = """Eres un experto en logística colombiana especializado en guías de transporte.
Analiza imágenes de guías de envío y extrae los datos con la mayor precisión posible.

Responde SIEMPRE con un JSON válido con estos campos:
{
  "tracking_code": "código de guía/rastreo",
  "carrier_name": "nombre de la transportadora",
  "tracking_url": "URL de rastreo si es visible, null si no",
  "recipient_name": "nombre COMPLETO del destinatario",
  "recipient_address": "dirección de entrega del destinatario",
  "recipient_city": "ciudad del destinatario",
  "confidence": 0.0-1.0
}

REGLAS:
- El nombre del destinatario y la dirección son CRÍTICOS para el matching con pedidos
- Si no puedes leer un valor, usar null, NUNCA inventar datos
- El tracking_code es el número de guía principal
- Reportar confidence 0.0-1.0 basado en la legibilidad de la imagen
- Responde SOLO con el JSON, sin texto adicional"""


def extract_guide_data(
    image_url: str,
    carrier_context: str = "",
) -> ExtractedGuideData:
    """
    Call Gemini Flash Vision to extract shipping guide data from an image.

    Args:
        image_url: Public URL of the shipping guide image
        carrier_context: Optional context about the carrier company

    Returns:
        ExtractedGuideData with the extracted fields
    """
    start = time.time()

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    user_prompt = f"Analiza esta imagen de guía de transporte colombiana.{carrier_context}\n\nExtrae los datos en el formato JSON especificado."

    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=image_url,
                            mime_type="image/jpeg",
                        ),
                        types.Part.from_text(text=user_prompt),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=2000,
                response_mime_type="application/json",
            ),
        )

        raw_text = response.text or "{}"
        elapsed = time.time() - start
        logger.info("Gemini vision completed in %.1fs", elapsed)

        # Parse JSON from response
        try:
            json_match = raw_text
            if not raw_text.strip().startswith("{"):
                import re
                match = re.search(r"\{[\s\S]*\}", raw_text)
                json_match = match.group(0) if match else "{}"

            data = json.loads(json_match)
            return ExtractedGuideData(**data)

        except (json.JSONDecodeError, ValueError) as parse_err:
            logger.error("Failed to parse Gemini response: %s", parse_err)
            logger.debug("Raw response: %s", raw_text[:500])
            return ExtractedGuideData()

    except Exception as exc:
        logger.error("Gemini Vision API error: %s", exc, exc_info=True)
        raise

"""
Agent 2: Conversation Analyzer — extracts purchase intents from chat messages
and internal notes. Internal notes have HIGHER PRIORITY than messages.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.config import settings
from app.models.extraction import ConversationOutput, CustomerUpdates, ItemRequested
from app.models.job import InternalNote, MessagePayload

logger = logging.getLogger(__name__)

# ── Prompt for conversation analysis ──────────────────────────

CONVERSATION_PROMPT = """Eres un analista experto de una óptica (Punto Gafas, Colombia).

Tu tarea es analizar la conversación entre un asesor y un cliente, junto con las notas internas del asesor, para extraer las intenciones de compra.

REGLA IMPORTANTE: Las NOTAS INTERNAS del asesor tienen MAYOR PESO que los mensajes del chat.
El asesor las escribe con información CONFIRMADA.

Extrae la información en el siguiente formato JSON:
{
  "items_requested": [
    {
      "type": "lente" | "montura" | "accesorio" | "servicio",
      "description": "descripción detallada del producto solicitado",
      "category": "progresivo" | "monofocal" | "bifocal" | "ocupacional" | null,
      "material_hint": "policarbonato" | "CR" | "trivex" | null,
      "treatment_hint": "transitions" | "blue block" | "antireflejo" | null,
      "is_digital": true | false | null,
      "brand_hint": "marca mencionada" | null,
      "model_hint": "modelo mencionado" | null,
      "quantity": 1,
      "notes": "notas adicionales relevantes"
    }
  ],
  "special_instructions": "instrucciones especiales del asesor o cliente" | null,
  "urgency": "normal" | "urgente" | "desconocida",
  "promised_date_hint": "YYYY-MM-DD" | null,
  "customer_updates": {
    "email": "email mencionado" | null,
    "document_id": "cédula/documento mencionado" | null,
    "city": "ciudad mencionada" | null,
    "phone": "teléfono nuevo" | null,
    "address": "dirección mencionada" | null
  }
}

REGLAS:
- Si se mencionan lentes, siempre especifica quantity=2 (par) a menos que se indique lo contrario
- "transitions" incluye: Transitions Signature, Transitions Gen8, fotocromáticos
- "blue block" incluye: blue light, filtro azul, blue/verde, blue UV
- Si el cliente menciona "progresivos", category="progresivo"
- Si menciona "lejos" o "cerca" sin más contexto, probablemente category="monofocal"
- Si menciona "lentes digitales", is_digital=true
- Si se menciona un descuento, inclúyelo en notes
- customer_updates: solo incluir datos explícitos que actualicen el registro del cliente
- Si no hay mensajes ni notas relevantes, retorna items_requested como array vacío
- Responde SOLO con el JSON, sin texto adicional"""


def _build_context(
    messages: list[MessagePayload],
    internal_notes: list[InternalNote],
    instructions: str | None,
) -> str:
    """Build a combined context string for the LLM."""
    parts: list[str] = []

    if internal_notes:
        parts.append("=== NOTAS INTERNAS DEL ASESOR (MAYOR PRIORIDAD) ===")
        for note in internal_notes:
            ts = f" [{note.created_at}]" if note.created_at else ""
            parts.append(f"📝{ts} {note.content or '(vacía)'}")

    if instructions:
        parts.append(f"\n=== INSTRUCCIONES ESPECIALES ===\n{instructions}")

    if messages:
        parts.append("\n=== CONVERSACIÓN DE CHAT ===")
        for msg in messages:
            role_label = "Cliente" if msg.role == "user" else "Asesor"
            ts = f" [{msg.created_at}]" if msg.created_at else ""
            content = msg.content or ""
            if msg.type and msg.type != "text":
                content += f" [Adjunto: {msg.type}]"
            parts.append(f"{role_label}{ts}: {content}")

    if not parts:
        return "(Sin mensajes ni notas internas disponibles)"

    return "\n".join(parts)


def _call_gemini_conversation(context: str) -> dict[str, Any]:
    """Send conversation context to Gemini for analysis."""
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[
                    genai_types.Content(
                        parts=[
                            genai_types.Part.from_text(CONVERSATION_PROMPT),
                            genai_types.Part.from_text(f"\n\n--- DATOS A ANALIZAR ---\n\n{context}"),
                        ]
                    )
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.2,
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
            logger.warning("Gemini conversation: invalid JSON (attempt %d): %s", attempt + 1, exc)
            if attempt == 2:
                return {"items_requested": [], "error": "Respuesta no parseable de IA"}

        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                delay = settings.RETRY_BASE_DELAY * (2 ** attempt)
                delay = min(delay, settings.RETRY_MAX_DELAY)
                logger.warning("Gemini rate limit (attempt %d), waiting %.1fs", attempt + 1, delay)
                time.sleep(delay)
            else:
                logger.error("Gemini conversation error (attempt %d): %s", attempt + 1, exc)
                if attempt == 2:
                    return {"items_requested": [], "error": str(exc)}

    return {"items_requested": [], "error": "Agotados reintentos de IA"}


def _parse_conversation_result(raw: dict[str, Any]) -> ConversationOutput:
    """Convert raw Gemini output to a typed ConversationOutput."""
    items = []
    for item_raw in raw.get("items_requested", []):
        items.append(ItemRequested(
            type=item_raw.get("type"),
            description=item_raw.get("description"),
            category=item_raw.get("category"),
            material_hint=item_raw.get("material_hint"),
            treatment_hint=item_raw.get("treatment_hint"),
            is_digital=item_raw.get("is_digital"),
            brand_hint=item_raw.get("brand_hint"),
            model_hint=item_raw.get("model_hint"),
            quantity=item_raw.get("quantity", 1),
            notes=item_raw.get("notes"),
        ))

    cu_raw = raw.get("customer_updates", {})
    customer_updates = None
    if cu_raw and any(v for v in cu_raw.values() if v):
        customer_updates = CustomerUpdates(
            email=cu_raw.get("email"),
            document_id=cu_raw.get("document_id"),
            city=cu_raw.get("city"),
            phone=cu_raw.get("phone"),
            address=cu_raw.get("address"),
        )

    warnings: list[str] = []
    if not items:
        warnings.append("No se identificaron productos en la conversación")
    if raw.get("error"):
        warnings.append(raw["error"])

    return ConversationOutput(
        items_requested=items,
        special_instructions=raw.get("special_instructions"),
        urgency=raw.get("urgency", "desconocida"),
        promised_date_hint=raw.get("promised_date_hint"),
        customer_updates=customer_updates,
        warnings=warnings,
        error=raw.get("error"),
    )


# ── Public API ────────────────────────────────────────────────

def run_conversation_analyzer(
    messages: list[MessagePayload],
    internal_notes: list[InternalNote],
    instructions: str | None = None,
) -> ConversationOutput:
    """
    Analyze conversation + notes to extract purchase intents.
    Always returns a valid ConversationOutput, even with no data.
    """
    has_messages = any(m.content for m in messages)
    has_notes = any(n.content for n in internal_notes)

    if not has_messages and not has_notes and not instructions:
        logger.warning("Conversation analyzer: no messages, notes, or instructions")
        return ConversationOutput(
            warnings=["Sin mensajes, notas internas ni instrucciones disponibles"],
        )

    context = _build_context(messages, internal_notes, instructions)
    logger.info("Conversation context length: %d chars", len(context))

    raw = _call_gemini_conversation(context)
    result = _parse_conversation_result(raw)

    logger.info(
        "Conversation analysis: %d items, urgency=%s",
        len(result.items_requested), result.urgency,
    )
    return result

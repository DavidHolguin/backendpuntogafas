"""
Agent 2: Conversation Analyzer — extracts purchase intents AND payment
mentions from chat messages and internal notes.
Internal notes have HIGHER PRIORITY than messages.

Sale-tag aware:
  - Includes sale_tag annotations in the LLM context
  - Detects venta_directa intent when all notes are tagged
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.config import settings
from app.models.extraction import (
    ConversationOutput,
    CustomerUpdates,
    ItemRequested,
    PaymentMention,
)
from app.models.job import InternalNote, MessagePayload

logger = logging.getLogger(__name__)

# ── Prompt for conversation analysis ──────────────────────────

CONVERSATION_PROMPT = """Eres un analista experto de una óptica (Punto Gafas, Colombia).

Tu tarea es analizar la conversación entre un asesor y un cliente, junto con las notas internas del asesor, para extraer las intenciones de compra Y detectar información sobre pagos.

REGLA IMPORTANTE: Las NOTAS INTERNAS del asesor tienen MAYOR PESO que los mensajes del chat.
El asesor las escribe con información CONFIRMADA.

ETIQUETAS DE VENTA (sale_tag):
- Si una nota tiene etiqueta [🏷️ montura], el asesor confirmó que es una venta de montura/armazón.
- Si una nota tiene etiqueta [🏷️ estuche], el asesor confirmó que es una venta de estuche.
- Cuando TODAS las notas tienen etiquetas de venta (montura y/o estuche) y NO hay fórmula óptica, 
  se trata de una VENTA DIRECTA de accesorios, no de un pedido óptico con lentes.
- En venta directa, los items deben ser tipo "montura" o "accesorio" según la etiqueta.

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
  },
  "payment_mentions": [
    {
      "method": "efectivo" | "transferencia" | "tarjeta" | "nequi" | "daviplata" | null,
      "type": "total" | "parcial" | null,
      "amount": <número o null>,
      "has_proof": true | false,
      "source": "conversation" | "internal_note",
      "raw_text": "fragmento original del mensaje"
    }
  ]
}

REGLAS PARA ITEMS:
- Si se mencionan lentes, siempre especifica quantity=2 (par) a menos que se indique lo contrario
- "transitions" incluye: Transitions Signature, Transitions Gen8, fotocromáticos
- "blue block" incluye: blue light, filtro azul, blue/verde, blue UV
- Si el cliente menciona "progresivos", category="progresivo"
- Si menciona "lejos" o "cerca" sin más contexto, probablemente category="monofocal"
- Si menciona "lentes digitales", is_digital=true
- Si se menciona un descuento, inclúyelo en notes
- customer_updates: solo incluir datos explícitos que actualicen el registro del cliente
- Si no hay mensajes ni notas relevantes, retorna items_requested como array vacío
- Para ventas directas (etiquetas montura/estuche): usa type="montura" para monturas y type="accesorio" para estuches, NO uses type="lente"

REGLAS PARA PAGOS (payment_mentions):
- Busca menciones de pago: "ya pagué", "hice transferencia", "le envío el comprobante", "pago con tarjeta"
- Busca montos: "le aboné 200 mil", "pago de 360.000", "pagó $160.000"
- Mapea método de pago:
  - "datáfono" / "datafono" = "tarjeta"
  - "nequi" = "nequi"
  - "daviplata" = "daviplata"
  - "efectivo" = "efectivo"
  - "transferencia" / "consignación" = "transferencia"
- Si un mensaje habla de pago Y tiene un adjunto de imagen = has_proof: true
- Si la mención viene de una nota interna, source="internal_note"
- Solo incluir payment_mentions si realmente se menciona un pago. NO inventar.
- Si no hay menciones de pago, retorna payment_mentions como array vacío

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
            tag = f" [🏷️ {note.sale_tag}]" if note.sale_tag else ""
            content = note.content or "(vacía)"
            if note.attachment_url:
                content += f" [Adjunto: {note.type or 'image'}]"
            parts.append(f"📝{ts}{tag} {content}")

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
            if msg.attachment_url:
                content += f" [URL adjunto: {msg.attachment_url}]"
            parts.append(f"{role_label}{ts}: {content}")

    if not parts:
        return "(Sin mensajes ni notas internas disponibles)"

    return "\n".join(parts)


def _detect_venta_directa(
    internal_notes: list[InternalNote],
    items: list[ItemRequested],
) -> str | None:
    """
    Detect if this is a venta_directa based on sale_tags and item types.

    Returns 'venta_directa' if:
      - There are internal notes with sale_tag set
      - All notes with content have sale_tag (montura or estuche)
      - No lens items were detected
    Otherwise returns 'optico' or None.
    """
    # Get notes that have actual content (not empty system notes)
    content_notes = [n for n in internal_notes if n.content]
    tagged_notes = [n for n in content_notes if n.sale_tag]

    if not tagged_notes:
        return None  # No tags → standard flow, let downstream decide

    # Check if ALL content notes are tagged
    all_tagged = len(tagged_notes) == len(content_notes) if content_notes else False

    # Check if any lens items were detected
    has_lens = any(item.type == "lente" for item in items)

    if all_tagged and not has_lens:
        return "venta_directa"

    # Even if some notes are tagged, if there's a lens request → optico
    if has_lens:
        return "optico"

    # Mixed scenario: some tagged, some not → optico (safer)
    return "optico"


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
                            genai_types.Part.from_text(text=CONVERSATION_PROMPT),
                            genai_types.Part.from_text(text=f"\n\n--- DATOS A ANALIZAR ---\n\n{context}"),
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
                return {"items_requested": [], "payment_mentions": [], "error": "Respuesta no parseable de IA"}

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
                    return {"items_requested": [], "payment_mentions": [], "error": str(exc)}

    return {"items_requested": [], "payment_mentions": [], "error": "Agotados reintentos de IA"}


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

    # Parse payment mentions
    payment_mentions = []
    for pm_raw in raw.get("payment_mentions", []):
        payment_mentions.append(PaymentMention(
            method=pm_raw.get("method"),
            type=pm_raw.get("type"),
            amount=float(pm_raw["amount"]) if pm_raw.get("amount") else None,
            has_proof=bool(pm_raw.get("has_proof", False)),
            proof_url=pm_raw.get("proof_url"),
            source=pm_raw.get("source", "conversation"),
            raw_text=pm_raw.get("raw_text"),
        ))

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
        payment_mentions=payment_mentions,
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
    Analyze conversation + notes to extract purchase intents and payment mentions.
    Detects venta_directa intent from sale_tags on internal notes.
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

    # ── Detect venta_directa from sale_tags ────────────────────
    suggested_order_type = _detect_venta_directa(internal_notes, result.items_requested)
    if suggested_order_type:
        result.suggested_order_type = suggested_order_type
        logger.info("Detected order type: %s", suggested_order_type)

    logger.info(
        "Conversation analysis: %d items, %d payment_mentions, urgency=%s, order_type=%s",
        len(result.items_requested),
        len(result.payment_mentions),
        result.urgency,
        result.suggested_order_type,
    )
    return result

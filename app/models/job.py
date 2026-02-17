"""
Pydantic models for the ai_order_jobs table and its JSONB payload.
Every field that comes from external data is Optional to enforce
the "always create a draft" philosophy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Payload sub-structures ────────────────────────────────────

class MessagePayload(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    type: Optional[str] = "text"
    attachment_url: Optional[str] = None
    created_at: Optional[str] = None


class InternalNote(BaseModel):
    content: Optional[str] = None
    type: Optional[str] = "text"           # "text" | "image"
    attachment_url: Optional[str] = None
    sale_tag: Optional[str] = None          # "montura" | "estuche" | None
    created_at: Optional[str] = None


class CustomerPayload(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    document_id: Optional[str] = None


class JobPayload(BaseModel):
    """The JSONB payload inside ai_order_jobs.payload."""
    conversation_id: Optional[str] = None
    customer: Optional[CustomerPayload] = None
    sede_id: Optional[str] = None
    messages: list[MessagePayload] = Field(default_factory=list)
    internal_notes: list[InternalNote] = Field(default_factory=list)
    media_urls: list[str] = Field(default_factory=list)
    instructions: Optional[str] = None
    incomplete: bool = False


# ── Full job row ──────────────────────────────────────────────

class AIOrderJob(BaseModel):
    """Represents a row from the ai_order_jobs table."""
    id: str
    conversation_id: str
    customer_id: str
    sede_id: str
    requested_by: str
    status: str = "pending"
    payload: JobPayload = Field(default_factory=JobPayload)
    result: Optional[dict[str, Any]] = None
    order_id: Optional[str] = None
    error_message: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

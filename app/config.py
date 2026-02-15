"""
Application configuration via environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configuration loaded from environment variables."""

    # ── Supabase ──────────────────────────────────────────────
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str

    # ── Gemini ────────────────────────────────────────────────
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # ── Worker ────────────────────────────────────────────────
    POLL_INTERVAL_SECONDS: int = 5
    JOB_TIMEOUT_SECONDS: int = 180
    MAX_RETRIES: int = 2

    # ── Server ────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── CrewAI ────────────────────────────────────────────────
    CREWAI_VERBOSE: bool = True

    # ── Retry backoff for Gemini rate-limits ───────────────────
    RETRY_BASE_DELAY: float = 1.0       # seconds
    RETRY_MAX_DELAY: float = 4.0        # seconds

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()  # type: ignore[call-arg]

"""
Supabase REST client using httpx — avoids the heavy supabase-py
dependency chain (realtime → pyroaring C extension).
Uses PostgREST API directly for all DB operations.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class SupabaseClient:
    """
    Lightweight Supabase client that wraps PostgREST REST API via httpx.
    Only implements the methods we actually need (select, insert, update).
    """

    def __init__(self, url: str, service_role_key: str):
        self.base_url = url.rstrip("/")
        self.rest_url = f"{self.base_url}/rest/v1"
        self.headers = {
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self._client = httpx.Client(
            timeout=30,
            headers=self.headers,
        )
        logger.info("Supabase REST client initialized → %s", self.base_url)

    # ── Table query builder ───────────────────────────────────

    def table(self, name: str) -> "TableQuery":
        return TableQuery(self, name)

    def close(self) -> None:
        self._client.close()


class TableQuery:
    """Chainable query builder that mimics supabase-py's fluent API."""

    def __init__(self, client: SupabaseClient, table: str):
        self._client = client
        self._table = table
        self._url = f"{client.rest_url}/{table}"
        self._params: dict[str, str] = {}
        self._method: str = "GET"
        self._body: Any = None
        self._headers: dict[str, str] = {}

    def select(self, columns: str = "*") -> "TableQuery":
        self._method = "GET"
        self._params["select"] = columns
        return self

    def insert(self, data: dict[str, Any] | list[dict[str, Any]]) -> "TableQuery":
        self._method = "POST"
        self._body = data
        return self

    def update(self, data: dict[str, Any]) -> "TableQuery":
        self._method = "PATCH"
        self._body = data
        return self

    def eq(self, column: str, value: Any) -> "TableQuery":
        self._params[column] = f"eq.{value}"
        return self

    def ilike(self, column: str, pattern: str) -> "TableQuery":
        self._params[column] = f"ilike.{pattern}"
        return self

    def order(self, column: str, desc: bool = False) -> "TableQuery":
        direction = "desc" if desc else "asc"
        self._params["order"] = f"{column}.{direction}"
        return self

    def limit(self, count: int) -> "TableQuery":
        self._params["limit"] = str(count)
        return self

    def execute(self) -> "QueryResult":
        """Execute the query and return the result."""
        try:
            if self._method == "GET":
                resp = self._client._client.get(self._url, params=self._params)
            elif self._method == "POST":
                resp = self._client._client.post(
                    self._url, json=self._body, params=self._params,
                )
            elif self._method == "PATCH":
                resp = self._client._client.patch(
                    self._url, json=self._body, params=self._params,
                )
            else:
                raise ValueError(f"Unsupported method: {self._method}")

            resp.raise_for_status()

            data = resp.json() if resp.content else []
            if isinstance(data, dict):
                data = [data]

            return QueryResult(data=data)

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Supabase %s %s failed [%d]: %s",
                self._method, self._table, exc.response.status_code,
                exc.response.text[:500],
            )
            raise
        except Exception as exc:
            logger.error("Supabase query error: %s", exc, exc_info=True)
            raise


class QueryResult:
    """Mimics supabase-py's response object."""

    def __init__(self, data: list[dict[str, Any]]):
        self.data = data


# ── Singleton ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_supabase() -> SupabaseClient:
    """Return the singleton Supabase REST client."""
    return SupabaseClient(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

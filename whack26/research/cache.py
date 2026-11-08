"""SQLite cache for research source responses."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_cache (
  cache_key TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  source TEXT NOT NULL,
  query TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  ttl_seconds INTEGER NOT NULL,
  status INTEGER NOT NULL,
  body TEXT NOT NULL,
  headers_json TEXT
);
CREATE INDEX IF NOT EXISTS api_cache_source ON api_cache(source);

CREATE TABLE IF NOT EXISTS research_queries (
  cache_key TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  queries_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS research_queries_dataset
  ON research_queries(dataset_id);
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def cache_key(
    dataset_id: str,
    source: str,
    query: str,
    params: Mapping[str, object] | None = None,
) -> str:
    """Build the stable cache key from dataset, source, query, and params."""
    payload = json.dumps(
        {
            "dataset_id": dataset_id,
            "source": source,
            "query": _normalize_query(query),
            "params": params or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedResponse:
    key: str
    dataset_id: str
    source: str
    query: str
    fetched_at: str
    ttl_seconds: int
    status: int
    body: str
    headers: dict[str, str]
    stale: bool = False


class ResearchCache:
    """Small response cache with TTL and offline behavior.

    Online mode treats an expired row as a miss. Offline mode returns expired
    rows so a demo can still complete from a previously warmed cache.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        offline: bool = False,
        refresh: bool = False,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.path = Path(path)
        self.offline = offline
        self.refresh = refresh
        self._now = now
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.stale_hits = 0
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ResearchCache":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "stale_hits": self.stale_hits,
            "hit_rate": round(self.hit_rate, 4),
        }

    def get(self, key: str) -> CachedResponse | None:
        if self.refresh and not self.offline:
            self.misses += 1
            return None
        row = self._connection.execute(
            "SELECT * FROM api_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        stale = self._is_expired(row["fetched_at"], row["ttl_seconds"])
        if stale and not self.offline:
            self.misses += 1
            return None
        self.hits += 1
        if stale:
            self.stale_hits += 1
        try:
            headers = json.loads(row["headers_json"] or "{}")
        except json.JSONDecodeError:
            headers = {}
        if not isinstance(headers, dict):
            headers = {}
        return CachedResponse(
            key=row["cache_key"],
            dataset_id=row["dataset_id"],
            source=row["source"],
            query=row["query"],
            fetched_at=row["fetched_at"],
            ttl_seconds=int(row["ttl_seconds"]),
            status=int(row["status"]),
            body=row["body"],
            headers={str(key): str(value) for key, value in headers.items()},
            stale=stale,
        )

    def put(
        self,
        key: str,
        *,
        dataset_id: str,
        source: str,
        query: str,
        status: int,
        body: str,
        ttl_seconds: int,
        headers: Mapping[str, str] | None = None,
        fetched_at: str | None = None,
    ) -> None:
        timestamp = fetched_at or self._now().strftime("%Y-%m-%dT%H:%M:%SZ")
        # Store the whole body. A truncated body is invalid JSON and would
        # either fail to parse on an offline warm cache or force a silent
        # refetch online; SQLite TEXT has no practical size limit here.
        stored_body = body
        self._connection.execute(
            """
            INSERT INTO api_cache (
              cache_key, dataset_id, source, query, fetched_at,
              ttl_seconds, status, body, headers_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
              dataset_id = excluded.dataset_id,
              source = excluded.source,
              query = excluded.query,
              fetched_at = excluded.fetched_at,
              ttl_seconds = excluded.ttl_seconds,
              status = excluded.status,
              body = excluded.body,
              headers_json = excluded.headers_json
            """,
            (
                key,
                dataset_id,
                source,
                _normalize_query(query),
                timestamp,
                int(ttl_seconds),
                int(status),
                stored_body,
                json.dumps(dict(headers or {}), sort_keys=True),
            ),
        )
        self._connection.commit()
        self.writes += 1

    def get_queries(self, key: str) -> list[dict[str, object]] | None:
        """Read generated queries so a warm run reuses the exact search terms.

        Query generation is nondeterministic. Without this cache, a second run
        would produce different cache keys and the per-source responses would
        all miss even though nothing upstream changed.
        """
        if self.refresh and not self.offline:
            return None
        row = self._connection.execute(
            "SELECT queries_json FROM research_queries WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        try:
            queries = json.loads(row["queries_json"])
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(queries, list):
            return None
        return [item for item in queries if isinstance(item, dict)]

    def put_queries(
        self,
        key: str,
        *,
        dataset_id: str,
        queries: list[Mapping[str, object]],
    ) -> None:
        timestamp = self._now().strftime("%Y-%m-%dT%H:%M:%SZ")
        self._connection.execute(
            """
            INSERT INTO research_queries (
              cache_key, dataset_id, created_at, queries_json
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
              dataset_id = excluded.dataset_id,
              created_at = excluded.created_at,
              queries_json = excluded.queries_json
            """,
            (
                key,
                dataset_id,
                timestamp,
                json.dumps(list(queries), sort_keys=True),
            ),
        )
        self._connection.commit()

    def _is_expired(self, fetched_at: str, ttl_seconds: int) -> bool:
        try:
            timestamp = datetime.fromisoformat(
                fetched_at.replace("Z", "+00:00")
            )
        except ValueError:
            return True
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age = (self._now() - timestamp).total_seconds()
        return age > int(ttl_seconds)

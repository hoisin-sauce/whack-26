"""Parse and retrieve chunks from the preloaded statistical-pitfalls corpus."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..embeddings import cosine

_CHUNK_ID_RE = re.compile(r"\*\*Chunk ID:\*\*\s*`([^`]+)`")
_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_RANKED_ID_RE = re.compile(r"^S(?:[1-9]|10)-")
_FALLBACK_CHUNKS = ["S1-assumptions", "S2-leakage", "S5-diagnostics"]


class CorpusError(RuntimeError):
    """Raised when the corpus file cannot be read or parsed."""


@dataclass
class CorpusChunk:
    id: str
    title: str
    text: str


@dataclass
class Corpus:
    path: Path
    version: str
    chunks: list[CorpusChunk]
    _vectors: dict[str, list[list[float]]] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> "Corpus":
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CorpusError(f"could not read corpus {path}: {exc}") from exc
        text = raw.decode("utf-8", errors="replace")
        digest = hashlib.sha256(raw).hexdigest()[:8]
        version = _frontmatter_version(text, digest)
        chunks = _parse_chunks(text)
        if not chunks:
            raise CorpusError("corpus contains no parseable chunks")
        return cls(path=path, version=version, chunks=chunks)

    def get(self, chunk_id: str) -> CorpusChunk | None:
        return next((chunk for chunk in self.chunks if chunk.id == chunk_id), None)

    def ranked_chunks(self) -> list[CorpusChunk]:
        return [chunk for chunk in self.chunks if _RANKED_ID_RE.match(chunk.id)]

    def retrieve(
        self,
        query: str,
        embeddings: Any,
        *,
        top_k: int = 4,
    ) -> tuple[list[CorpusChunk], list[str]]:
        """Always include S0; rank the ten numbered sections by cosine."""
        warnings: list[str] = []
        selected: list[CorpusChunk] = []
        how_to_use = self.get("S0-how-to-use")
        if how_to_use is not None:
            selected.append(how_to_use)
        else:
            warnings.append("critic.corpus_s0_missing")

        ranked = self.ranked_chunks()
        if not ranked:
            fallback = [
                chunk
                for chunk in (self.get(item) for item in _FALLBACK_CHUNKS)
                if chunk is not None
            ]
            return selected + fallback[:top_k], warnings

        try:
            key = getattr(embeddings, "model", None) or str(id(embeddings))
            if key not in self._vectors:
                vectors = embeddings.embed([query] + [chunk.text for chunk in ranked])
                if len(vectors) != len(ranked) + 1:
                    raise ValueError("embedding count mismatch")
                self._vectors[key] = vectors
            vectors = self._vectors[key]
            query_vector = vectors[0]
            scored = [
                (chunk, cosine(query_vector, vector))
                for chunk, vector in zip(ranked, vectors[1:])
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            selected.extend(chunk for chunk, _ in scored[:top_k])
        except Exception as exc:  # noqa: BLE001 - retrieval must not block the run
            warnings.append(
                f"critic.retrieval_failed: {type(exc).__name__}: {exc}"
            )
            selected.extend(
                chunk
                for chunk in (self.get(item) for item in _FALLBACK_CHUNKS)
                if chunk is not None
            )
        return selected[: top_k + 1], warnings


def _frontmatter_version(text: str, digest: str) -> str:
    match = re.search(
        r"^---\s*$(.*?)^---\s*$", text, re.MULTILINE | re.DOTALL
    )
    if match:
        version_match = re.search(
            r"^version:\s*[\"']?([^\s\"']+)", match.group(1), re.MULTILINE
        )
        if version_match:
            return f"{version_match.group(1)}+{digest}"
    return f"unknown+{digest}"


def _parse_chunks(text: str) -> list[CorpusChunk]:
    matches = list(_SECTION_RE.finditer(text))
    chunks: list[CorpusChunk] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        id_match = _CHUNK_ID_RE.search(body)
        if id_match is None:
            continue
        chunk_id = id_match.group(1).strip()
        title = match.group(1).strip()
        chunks.append(CorpusChunk(id=chunk_id, title=title, text=body))
    return chunks


def load_corpus(path: Path) -> Corpus | None:
    """Return the corpus, or None when the file is absent."""
    if not path.is_file():
        return None
    return Corpus.from_path(path)

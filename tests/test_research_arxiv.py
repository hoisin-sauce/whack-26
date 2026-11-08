from __future__ import annotations

from pathlib import Path
from typing import Callable

import httpx
import pytest

from whack26.research.arxiv import (
    ARXIV_NAMESPACE,
    ATOM_NAMESPACE,
    arXivAdapter,
    reset_rate_limit,
)
from whack26.research.base import SourceError
from whack26.research.cache import ResearchCache


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _entry(
    entry_id: str = "http://arxiv.org/abs/2401.12345v1",
    *,
    title: str = "Housing prices and income",
    summary: str = "We study housing prices.",
    published: str | None = "2024-01-24T10:00:00Z",
    authors: tuple[str, ...] = ("Ada Lovelace", "Grace Hopper"),
    doi: str | None = "10.1234/abcd",
    alternate: str | None = "https://arxiv.org/abs/2401.12345v1",
) -> str:
    author_xml = "".join(
        f"<author><name>{author}</name></author>" for author in authors
    )
    doi_xml = f"<arxiv:doi>{doi}</arxiv:doi>" if doi else ""
    published_xml = f"<published>{published}</published>" if published else ""
    link_xml = (
        f'<link rel="alternate" href="{alternate}"/>' if alternate else ""
    )
    return (
        "<entry>"
        f"<id>{entry_id}</id>"
        f"{published_xml}"
        f"<title>{title}</title>"
        f"<summary>{summary}</summary>"
        f"{author_xml}"
        '<link rel="related" href="https://example.org/related"/>'
        f"{link_xml}{doi_xml}"
        "</entry>"
    )


def _feed(*entries: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<feed xmlns="{ATOM_NAMESPACE}" xmlns:arxiv="{ARXIV_NAMESPACE}">'
        f"{''.join(entries)}"
        "</feed>"
    )


@pytest.fixture(autouse=True)
def _isolate_rate_limit():
    reset_rate_limit()
    yield
    reset_rate_limit()


def test_search_parses_atom_and_sends_expected_params() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url
        captured["accept"] = request.headers.get("accept")
        return httpx.Response(
            200,
            text=_feed(
                _entry(),
                _entry(
                    entry_id="http://arxiv.org/abs/hep-th/9901001v1",
                    title="  Old style identifiers  ",
                    summary="Older work.",
                    published="1999-01-01T00:00:00Z",
                    authors=("Alan Turing",),
                    doi=None,
                    alternate=None,
                ),
                _entry(entry_id="http://arxiv.org/abs/2501.99999v1", title="  "),
            ),
        )

    adapter = arXivAdapter(sleep=lambda _: None)
    assert adapter.name == "arxiv"
    assert adapter.ttl_seconds == 24 * 60 * 60

    with _client(handler) as client:
        records = adapter.search("housing prices", 5, client)

    url = captured["url"]
    assert isinstance(url, httpx.URL)
    assert str(url).startswith("https://export.arxiv.org/api/query?")
    assert url.params["search_query"] == "all:housing prices"
    assert url.params["start"] == "0"
    assert url.params["max_results"] == "5"
    assert url.params["sortBy"] == "relevance"
    assert captured["accept"] == "application/atom+xml"

    assert len(records) == 2
    first = records[0]
    assert first.id == "arxiv:2401.12345v1"
    assert first.source == "arxiv"
    assert first.source_id == "2401.12345v1"
    assert first.title == "Housing prices and income"
    assert first.abstract == "We study housing prices."
    assert first.authors == ["Ada Lovelace", "Grace Hopper"]
    assert first.year == 2024
    assert first.doi == "10.1234/abcd"
    assert first.url == "https://arxiv.org/abs/2401.12345v1"
    assert first.license is None
    assert first.query == "housing prices"
    assert first.retrieved_at

    second = records[1]
    assert second.id == "arxiv:hep-th/9901001v1"
    assert second.source_id == "hep-th/9901001v1"
    assert second.title == "Old style identifiers"
    assert second.authors == ["Alan Turing"]
    assert second.year == 1999
    assert second.doi is None
    assert second.url == "http://arxiv.org/abs/hep-th/9901001v1"


def test_search_replays_from_cache(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research.db")
    adapter = arXivAdapter(
        cache=cache, dataset_id="ds_a", sleep=lambda _: None
    )
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, text=_feed(_entry()))

    with _client(handler) as client:
        first = adapter.search("housing", 5, client)
        second = adapter.search("housing", 5, client)

    assert calls["count"] == 1
    assert len(first) == len(second) == 1
    assert first[0].id == "arxiv:2401.12345v1"
    assert cache.stats()["hits"] == 1
    cache.close()


def test_rate_limit_is_global_across_adapters() -> None:
    sleeps: list[float] = []
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, text=_feed(_entry()))

    first = arXivAdapter(sleep=sleeps.append)
    second = arXivAdapter(sleep=sleeps.append)
    assert first.min_interval == 3.0

    with _client(handler) as client:
        first.search("housing", 5, client)
        second.search("income", 5, client)

    assert calls["count"] == 2
    assert len(sleeps) == 1
    assert sleeps[0] >= 3.0


def test_malformed_xml_raises_source_error() -> None:
    adapter = arXivAdapter(sleep=lambda _: None)

    with pytest.raises(SourceError, match="Atom") as exc_info:
        adapter.parse("<feed><entry>", query="housing")

    assert exc_info.value.source == "arxiv"
    assert exc_info.value.query == "housing"


def test_transient_503_retries_then_succeeds() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text=_feed(_entry()))

    adapter = arXivAdapter(sleep=sleeps.append)
    with _client(handler) as client:
        records = adapter.search("housing", 5, client)

    assert calls["count"] == 2
    assert len(records) == 1
    assert sleeps == [1.0]

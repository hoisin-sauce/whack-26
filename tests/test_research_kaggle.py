from __future__ import annotations

from typing import get_args

import httpx
import pytest

from whack26.research import kaggle as kaggle_module
from whack26.research.base import DATASET_SOURCES, SourceError, SourceName
from whack26.research.kaggle import KAGGLE_DATASETS_URL, KaggleAdapter
from whack26.research.normalize import deduplicate, record_identity_keys


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    real = kaggle_module.request_with_retries

    def wrapper(*args, **kwargs):
        kwargs.setdefault("sleep", lambda _seconds: None)
        return real(*args, **kwargs)

    monkeypatch.setattr(kaggle_module, "request_with_retries", wrapper)


def test_kaggle_search_maps_fields_and_sends_bearer_token() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json=[
                {
                    "ref": "owner/housing",
                    "title": "Housing Prices",
                    "description": "<p>California <b>housing</b> data</p>",
                    "subtitle": "Prices",
                    "licenseName": "CC0: Public Domain",
                    "ownerName": "Ada Lovelace",
                    "lastUpdated": "2021-06-01T12:00:00Z",
                    "url": "https://www.kaggle.com/datasets/owner/housing",
                    "isPrivate": False,
                }
            ],
        )

    adapter = KaggleAdapter(token="KGAT_test", username="ada")
    assert adapter.name == "kaggle"
    assert adapter.ttl_seconds == 24 * 60 * 60
    assert adapter.limited_mode is False

    with _client(handler) as client:
        records = adapter.search("Housing Prices", 5, client)

    url = seen["url"]
    assert isinstance(url, httpx.URL)
    assert str(url).startswith(KAGGLE_DATASETS_URL)
    assert url.params["search"] == "Housing Prices"
    assert url.params["page"] == "1"
    assert seen["authorization"] == "Bearer KGAT_test"

    assert len(records) == 1
    record = records[0]
    assert record.id == "kaggle:owner/housing"
    assert record.source == "kaggle"
    assert record.source_id == "owner/housing"
    assert record.title == "Housing Prices"
    assert record.abstract == "California housing data"
    assert record.authors == ["Ada Lovelace"]
    assert record.year == 2021
    assert record.doi is None
    assert record.url == "https://www.kaggle.com/datasets/owner/housing"
    assert record.license == "CC0: Public Domain"
    assert record.query == "Housing Prices"


def test_kaggle_reads_environment_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_env")
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")

    adapter = KaggleAdapter()

    assert adapter.token == "KGAT_env"
    assert adapter.username == "env-user"
    assert adapter.limited_mode is False


def test_kaggle_keyless_returns_empty_limited_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("keyless Kaggle must not make an HTTP request")

    adapter = KaggleAdapter()
    assert adapter.limited_mode is True
    with _client(handler) as client:
        records = adapter.search("housing", 5, client)

    assert records == []
    assert len(adapter.warnings) == 1
    assert "KAGGLE_API_TOKEN" in adapter.warnings[0]


def test_kaggle_dedups_duplicate_titles_and_skips_private() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "ref": "owner/housing",
                    "title": "Housing Prices",
                    "description": "Housing data",
                    "ownerName": "Ada Lovelace",
                    "lastUpdated": "2021-06-01T12:00:00Z",
                },
                {
                    "ref": "other/housing-copy",
                    "title": "Housing Prices",
                    "ownerName": "Ada Lovelace",
                    "lastUpdated": "2021-09-01T12:00:00Z",
                },
                {
                    "ref": "secret/data",
                    "title": "Private data",
                    "isPrivate": True,
                },
                {"ref": "missing-title", "title": "   "},
            ],
        )

    adapter = KaggleAdapter(token="KGAT_test")
    with _client(handler) as client:
        records = adapter.search("housing", 10, client)

    assert [record.source_id for record in records] == [
        "owner/housing",
        "other/housing-copy",
    ]
    keys = record_identity_keys(records[0])
    assert any(key.startswith("title:") for key in keys)

    merged = deduplicate(records)
    assert len(merged) == 1
    assert merged[0].source_id == "owner/housing"


def test_kaggle_trims_oversized_page_to_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"ref": f"owner/data-{index}", "title": f"Data {index}"}
                for index in range(20)
            ],
        )

    adapter = KaggleAdapter(token="KGAT_test")
    with _client(handler) as client:
        records = adapter.search("data", 3, client)

    assert len(records) == 3
    assert records[0].title == "Data 0"


def test_kaggle_auth_failure_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid token")

    adapter = KaggleAdapter(token="bad-token")
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("housing", 5, client)

    assert exc_info.value.source == "kaggle"
    assert exc_info.value.query == "housing"
    assert exc_info.value.status_code == 401


def test_kaggle_rate_limit_raises_source_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fast_retries(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    adapter = KaggleAdapter(token="KGAT_test")
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("housing", 5, client)

    assert exc_info.value.status_code == 429


def test_kaggle_parse_error_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    adapter = KaggleAdapter(token="KGAT_test")
    with _client(handler) as client:
        with pytest.raises(SourceError, match="expected a JSON list"):
            adapter.search("housing", 5, client)


def test_kaggle_source_name_is_registered() -> None:
    assert KaggleAdapter.name == "kaggle"
    assert "kaggle" in get_args(SourceName)
    assert "kaggle" in DATASET_SOURCES

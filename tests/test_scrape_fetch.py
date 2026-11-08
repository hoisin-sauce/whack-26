from __future__ import annotations

import threading
import time

import pytest
from flask import Flask, Response, redirect, request
from werkzeug.serving import make_server

from whack26.config import Settings
from whack26.embeddings import MockEmbeddingClient
from whack26.llm import MockLLMClient
from whack26.research.scrape.fetch import (
    DomainRateLimiter,
    Fetcher,
    parse_robots,
    robots_allowed,
)
from whack26.research.scrape.identity import url_hash
from whack26.research.scrape.links import LinkCandidate
from whack26.research.scrape.models import Document
from whack26.research.scrape.pipeline import run_semantic_scraping
from whack26.store import FieldStore

ROBOTS = """User-agent: *
Disallow: /private
Allow: /public
Crawl-delay: 0
"""


class MockState:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str | None]] = []
        self.cookie_headers: list[tuple[str, str]] = []
        self.robots_status = 200
        self.robots_text = ROBOTS
        self.conditional_304 = False
        self.etag = '"page-v1"'
        self.slow_seconds = 1.0
        self.large_size = 20_000
        self.lock = threading.Lock()

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()
            self.cookie_headers.clear()
        self.robots_status = 200
        self.robots_text = ROBOTS
        self.conditional_304 = False
        self.etag = '"page-v1"'
        self.slow_seconds = 1.0
        self.large_size = 20_000


STATE = MockState()
app = Flask(__name__)
app.logger.disabled = True


@app.before_request
def _record_request() -> None:
    with STATE.lock:
        STATE.requests.append(
            (
                request.path,
                request.headers.get("User-Agent", ""),
                request.headers.get("If-None-Match"),
            )
        )
        STATE.cookie_headers.append(
            (request.path, request.headers.get("Cookie", ""))
        )


@app.route("/robots.txt")
def robots() -> Response:
    if STATE.robots_status != 200:
        return Response("", status=STATE.robots_status)
    return Response(STATE.robots_text, mimetype="text/plain")


@app.route("/public/page")
def public_page() -> Response:
    if STATE.conditional_304 and request.headers.get("If-None-Match") == STATE.etag:
        return Response("", status=304, headers={"ETag": STATE.etag})
    html = (
        "<html><head><title>Public Page</title>"
        '<meta name="description" content="A public page."></head>'
        "<body><p>Hello public world.</p>"
        '<a href="/public/child">child</a>'
        '<a href="/private/blocked">private</a>'
        '<a href="/file.pdf">pdf</a>'
        '<a href="https://example.test/external">external</a>'
        "</body></html>"
    )
    return Response(html, mimetype="text/html", headers={"ETag": STATE.etag})


@app.route("/public/child")
def public_child() -> Response:
    return Response(
        "<html><title>Child Page</title><body>Child text.</body></html>",
        mimetype="text/html",
    )


@app.route("/private/blocked")
def private_blocked() -> Response:
    return Response("should not be fetched", mimetype="text/html")


@app.route("/file.pdf")
def pdf() -> Response:
    return Response(b"%PDF-1.4 fake", mimetype="application/pdf")


@app.route("/binary")
def binary() -> Response:
    return Response(b"\x00\x01\x02", mimetype="application/octet-stream")


@app.route("/redirect")
def redirect_public() -> Response:
    return redirect("/public/page", code=302)


@app.route("/redirect-private")
def redirect_private() -> Response:
    return redirect("/private/blocked", code=302)


@app.route("/slow")
def slow() -> Response:
    time.sleep(STATE.slow_seconds)
    return Response("slow", mimetype="text/html")


@app.route("/too-large")
def too_large() -> Response:
    return Response("x" * STATE.large_size, mimetype="text/html")


@app.route("/json")
def json_page() -> Response:
    return Response('{"ok": true}', mimetype="application/json")


@app.route("/set-cookie")
def set_cookie() -> Response:
    response = Response(
        "<html><title>Cookie Page</title><body>cookie</body></html>",
        mimetype="text/html",
    )
    response.set_cookie("session", "secret")
    return response


@app.route("/nofollow")
def nofollow() -> Response:
    return Response(
        '<html><head><meta name="robots" content="nofollow"></head>'
        '<body><a href="/public/child">child</a></body></html>',
        mimetype="text/html",
    )


@app.route("/noarchive")
def noarchive() -> Response:
    return Response(
        '<html><head><meta name="robots" content="noarchive"></head>'
        "<body>no archive</body></html>",
        mimetype="text/html",
    )


@app.route("/x-robots-noarchive")
def x_robots_noarchive() -> Response:
    return Response(
        "<html><body>no archive header</body></html>",
        mimetype="text/html",
        headers={"X-Robots-Tag": "noarchive"},
    )


@app.route("/download.pdf")
def download_pdf() -> Response:
    return Response("<html><title>Not really a PDF</title></html>", mimetype="text/html")


@app.route("/missing")
def missing() -> Response:
    return Response("missing", status=404, mimetype="text/html")


@app.route("/status/429")
def status_429() -> Response:
    return Response("rate limited", status=429, headers={"Retry-After": "60"})


@pytest.fixture(scope="module")
def server() -> str:
    httpd = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def reset_state() -> None:
    STATE.reset()


def make_settings(**overrides: object) -> Settings:
    base = Settings.from_env({}, mock=True)
    values = {
        "scrape_fetch_enabled": True,
        "scrape_allow_private_networks": True,
        "scrape_rate_limit_seconds": 0.0,
        "scrape_fetch_timeout": 1.0,
        "scrape_max_bytes": 10_000,
        "scrape_max_text_chars": 500,
        "scrape_max_redirects": 2,
    }
    values.update(overrides)
    return base.with_overrides(**values)


def make_fetcher(
    server: str,
    *,
    settings: Settings | None = None,
    clock: object | None = None,
    sleep: object | None = None,
) -> tuple[Fetcher, FieldStore]:
    store = FieldStore(":memory:", dims=2)
    fetcher = Fetcher(
        settings=settings or make_settings(),
        store=store,
        allow_private_networks=True,
        clock=clock or time.monotonic,
        sleep=sleep or time.sleep,
    )
    return fetcher, store


def request_paths() -> list[str]:
    return [path for path, _, _ in STATE.requests]


def test_parse_robots_longest_match_and_tie_allow() -> None:
    rules = parse_robots(
        "User-agent: *\nDisallow: /private\nAllow: /private/public\n",
        "whack26-research/0.1",
    )
    assert robots_allowed(rules, "/private/page") is False
    assert robots_allowed(rules, "/private/public/page") is True
    assert robots_allowed(rules, "/other") is True


def test_parse_robots_crawl_delay_and_empty_disallow() -> None:
    rules = parse_robots(
        "User-agent: *\nDisallow:\nCrawl-delay: 2.5\n", "whack26-research/0.1"
    )
    assert rules.allow_all is True
    assert rules.crawl_delay == pytest.approx(2.5)


def test_parse_robots_specific_agent_overrides_star() -> None:
    rules = parse_robots(
        "User-agent: *\nAllow: /\n\n"
        "User-agent: whack26-research\nDisallow: /private\n",
        "whack26-research/0.1",
    )
    assert robots_allowed(rules, "/private/page") is False
    assert robots_allowed(rules, "/public/page") is True


def test_domain_rate_limiter_uses_injected_clock() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = DomainRateLimiter(1.0, clock=clock, sleep=sleep)
    limiter.wait("example.test")
    limiter.wait("example.test")
    limiter.wait("example.test")

    assert sleeps == [pytest.approx(1.0), pytest.approx(1.0)]


def test_fetch_public_page_extracts_title_text_and_links(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        result = fetcher.fetch(server + "/public/page")
        assert result is not None
        assert result.status == 200
        assert result.title == "Public Page"
        assert "Hello public world." in result.text
        assert any(link.endswith("/public/child") for link in result.links)
        assert any(link.endswith("/private/blocked") for link in result.links)
        assert any(link.endswith("/file.pdf") for link in result.links)
        assert result.content_sha512
        assert result.from_cache is False
    finally:
        fetcher.close()
        store.close()


def test_fetch_robots_disallow_blocks_without_request(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        result = fetcher.fetch(server + "/private/blocked")
        assert result is None
        assert "/private/blocked" not in request_paths()
        assert any("fetch_robots_disallowed" in warning for warning in fetcher.warnings)
    finally:
        fetcher.close()
        store.close()


def test_fetch_robots_allow_override(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        result = fetcher.fetch(server + "/public/child")
        assert result is not None
        assert result.title == "Child Page"
    finally:
        fetcher.close()
        store.close()


def test_fetch_rejects_pdf_without_caching(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        result = fetcher.fetch(server + "/file.pdf")
        assert result is None
        assert any(
            "fetch_pdf_rejected" in warning
            or "fetch_content_type_rejected" in warning
            for warning in fetcher.warnings
        )
        assert store.get_fetch(url_hash(server + "/file.pdf")) is None
    finally:
        fetcher.close()
        store.close()


def test_fetch_rejects_binary(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        assert fetcher.fetch(server + "/binary") is None
        assert any("fetch_content_type_rejected" in warning for warning in fetcher.warnings)
    finally:
        fetcher.close()
        store.close()


def test_fetch_rejects_pdf_suffix_even_with_html_type(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        assert fetcher.fetch(server + "/download.pdf") is None
        assert any("fetch_pdf_rejected" in warning for warning in fetcher.warnings)
    finally:
        fetcher.close()
        store.close()


def test_fetch_meta_nofollow_skips_links(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        result = fetcher.fetch(server + "/nofollow")
        assert result is not None
        assert result.links == []
    finally:
        fetcher.close()
        store.close()


def test_fetch_meta_noarchive_skips_cache(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        assert fetcher.fetch(server + "/noarchive") is not None
        assert store.get_fetch(url_hash(server + "/noarchive")) is None
    finally:
        fetcher.close()
        store.close()


def test_fetch_x_robots_noarchive_skips_cache(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        assert fetcher.fetch(server + "/x-robots-noarchive") is not None
        assert store.get_fetch(url_hash(server + "/x-robots-noarchive")) is None
    finally:
        fetcher.close()
        store.close()


def test_fetch_follows_redirect_and_rechecks_robots(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        result = fetcher.fetch(server + "/redirect")
        assert result is not None
        assert result.final_url.endswith("/public/page")
        assert "/redirect" in request_paths()
        assert "/public/page" in request_paths()

        STATE.requests.clear()
        blocked = fetcher.fetch(server + "/redirect-private")
        assert blocked is None
        assert "/redirect-private" in request_paths()
        assert "/private/blocked" not in request_paths()
    finally:
        fetcher.close()
        store.close()


def test_fetch_redirect_limit(server: str) -> None:
    fetcher, store = make_fetcher(
        server, settings=make_settings(scrape_max_redirects=0)
    )
    try:
        assert fetcher.fetch(server + "/redirect") is None
        assert any("fetch_too_many_redirects" in warning for warning in fetcher.warnings)
    finally:
        fetcher.close()
        store.close()


def test_fetch_timeout(server: str) -> None:
    STATE.slow_seconds = 0.5
    fetcher, store = make_fetcher(
        server, settings=make_settings(scrape_fetch_timeout=0.1)
    )
    try:
        assert fetcher.fetch(server + "/slow") is None
        assert any("fetch_timeout" in warning for warning in fetcher.warnings)
    finally:
        fetcher.close()
        store.close()


def test_fetch_size_cap(server: str) -> None:
    STATE.large_size = 20_000
    fetcher, store = make_fetcher(
        server, settings=make_settings(scrape_max_bytes=1_000)
    )
    try:
        assert fetcher.fetch(server + "/too-large") is None
        assert any("fetch_too_large" in warning for warning in fetcher.warnings)
    finally:
        fetcher.close()
        store.close()


def test_fetch_cache_hit_makes_one_page_request(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        first = fetcher.fetch(server + "/public/page")
        second = fetcher.fetch(server + "/public/page")
        assert first is not None and second is not None
        assert second.from_cache is True
        assert request_paths().count("/public/page") == 1
    finally:
        fetcher.close()
        store.close()


def test_fetch_conditional_304_reuses_cache(server: str) -> None:
    STATE.conditional_304 = True
    fetcher, store = make_fetcher(server)
    try:
        first = fetcher.fetch(server + "/public/page")
        second = fetcher.fetch(server + "/public/page", use_cache=False)
        assert first is not None and second is not None
        assert second.from_cache is True
        assert request_paths().count("/public/page") == 2
        assert any(
            header == STATE.etag
            for _, _, header in STATE.requests
            if header is not None
        )
    finally:
        fetcher.close()
        store.close()


def test_fetch_blocks_private_networks_by_default(server: str) -> None:
    store = FieldStore(":memory:", dims=2)
    fetcher = Fetcher(
        settings=make_settings(scrape_allow_private_networks=False),
        store=store,
        allow_private_networks=False,
    )
    try:
        assert fetcher.fetch(server + "/public/page") is None
        assert any(
            "fetch_private_network_blocked" in warning for warning in fetcher.warnings
        )
        assert request_paths() == []
    finally:
        fetcher.close()
        store.close()


def test_fetch_domain_allowlist_and_blocklist(server: str) -> None:
    store = FieldStore(":memory:", dims=2)
    allow = Fetcher(
        settings=make_settings(scrape_allowed_domains=("example.com",)),
        store=store,
        allow_private_networks=True,
    )
    try:
        assert allow.fetch(server + "/public/page") is None
        assert any("fetch_domain_blocked" in warning for warning in allow.warnings)
    finally:
        allow.close()
    block = Fetcher(
        settings=make_settings(scrape_blocked_domains=("127.0.0.1",)),
        store=store,
        allow_private_networks=True,
    )
    try:
        assert block.fetch(server + "/public/page") is None
        assert any("fetch_domain_blocked" in warning for warning in block.warnings)
    finally:
        block.close()
        store.close()


def test_fetch_robots_5xx_disallows(server: str) -> None:
    STATE.robots_status = 500
    fetcher, store = make_fetcher(server)
    try:
        assert fetcher.fetch(server + "/public/page") is None
        assert any("robots_status_500" in warning for warning in fetcher.warnings)
        assert "/public/page" not in request_paths()
    finally:
        fetcher.close()
        store.close()


def test_fetch_robots_404_allows(server: str) -> None:
    STATE.robots_status = 404
    fetcher, store = make_fetcher(server)
    try:
        assert fetcher.fetch(server + "/public/page") is not None
    finally:
        fetcher.close()
        store.close()


def test_fetch_robots_403_disallows(server: str) -> None:
    STATE.robots_status = 403
    fetcher, store = make_fetcher(server)
    try:
        assert fetcher.fetch(server + "/public/page") is None
        assert any(
            "fetch_robots_disallowed" in warning for warning in fetcher.warnings
        )
        assert "/public/page" not in request_paths()
    finally:
        fetcher.close()
        store.close()


def test_fetch_robots_redirect_disallows(server: str) -> None:
    STATE.robots_status = 302
    fetcher, store = make_fetcher(server)
    try:
        assert fetcher.fetch(server + "/public/page") is None
        assert any("robots_redirect" in warning for warning in fetcher.warnings)
    finally:
        fetcher.close()
        store.close()


def test_fetch_json_content(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        result = fetcher.fetch(server + "/json")
        assert result is not None
        assert result.content_type.startswith("application/json")
        assert '"ok": true' in result.text
    finally:
        fetcher.close()
        store.close()


def test_fetch_404_and_429_are_warnings(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        assert fetcher.fetch(server + "/missing") is None
        assert any("fetch_status_404" in warning for warning in fetcher.warnings)
        assert fetcher.fetch(server + "/status/429") is None
        assert any("fetch_status_429" in warning for warning in fetcher.warnings)
    finally:
        fetcher.close()
        store.close()


def test_fetch_document_builds_web_document(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        candidate = LinkCandidate(
            url=server + "/public/page",
            title="Candidate title",
            relation="references",
        )
        document = fetcher.fetch_document(candidate)
        assert document is not None
        assert document.kind == "web"
        assert document.source == "web"
        assert document.status == "fetched"
        assert document.canonical_url == server + "/public/page"
        assert document.metadata["public_text"]
        assert document.metadata["links"]
        assert document.content_sha512
    finally:
        fetcher.close()
        store.close()


def test_fetch_text_is_capped_and_not_full_html(server: str) -> None:
    fetcher, store = make_fetcher(
        server, settings=make_settings(scrape_max_text_chars=30)
    )
    try:
        result = fetcher.fetch(server + "/public/page")
        assert result is not None
        assert len(result.text) <= 30
        assert "[truncated]" in result.text
        row = store.get_fetch(url_hash(server + "/public/page"))
        assert row is not None
        assert "<html" not in row["body"].lower()
        assert len(row["body"]) <= 30
    finally:
        fetcher.close()
        store.close()


def test_fetch_sends_user_agent(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        fetcher.fetch(server + "/public/page")
        agents = [agent for _, agent, _ in STATE.requests]
        assert any("whack26-research" in agent for agent in agents)
    finally:
        fetcher.close()
        store.close()


def test_fetch_does_not_send_cookies(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        assert fetcher.fetch(server + "/set-cookie") is not None
        assert fetcher.fetch(server + "/public/page") is not None
        page_cookies = [
            cookie
            for path, cookie in STATE.cookie_headers
            if path == "/public/page"
        ]
        assert page_cookies == [""]
    finally:
        fetcher.close()
        store.close()


def test_robots_is_fetched_once_per_origin(server: str) -> None:
    fetcher, store = make_fetcher(server)
    try:
        fetcher.fetch(server + "/public/page")
        fetcher.fetch(server + "/public/child")
        assert request_paths().count("/robots.txt") == 1
    finally:
        fetcher.close()
        store.close()


def test_crawl_delay_is_respected_with_fake_clock(server: str) -> None:
    STATE.robots_text = "User-agent: *\nAllow: /\nCrawl-delay: 2\n"
    now = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    fetcher, store = make_fetcher(server, clock=clock, sleep=sleep)
    try:
        fetcher.fetch(server + "/public/page")
        fetcher.fetch(server + "/public/child")
        assert any(seconds >= 2.0 for seconds in sleeps)
    finally:
        fetcher.close()
        store.close()


def test_pipeline_uses_compliant_fetcher(server: str) -> None:
    settings = make_settings(
        embed_dims=16,
        scrape_enabled=True,
        scrape_summary_threshold=0.0,
        scrape_link_threshold=0.0,
        scrape_max_resources=5,
        scrape_max_depth=1,
    )
    store = FieldStore(":memory:", dims=16)
    seed = Document(
        id="doc_seed",
        kind="paper",
        source="openalex",
        title="Seed paper",
        summary="Seed summary.",
        summary_embedding=[1.0] + [0.0] * 15,
        embedding_model="mock@16",
        metadata={
            "links": [server + "/public/page"],
            "abstract": "Seed abstract.",
        },
        status="summarized",
    )
    llm = MockLLMClient()

    class FixedEmbeddings:
        def __init__(self, dims: int = 16) -> None:
            self.dims = dims

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * (self.dims - 1) for _ in texts]

    embeddings = FixedEmbeddings(16)
    try:
        result = run_semantic_scraping(
            seeds=[seed],
            target="housing prices",
            store=store,
            llm=llm,
            embeddings=embeddings,
            settings=settings,
        )
        web = [document for document in result.documents if document.kind == "web"]
        assert web, result.warnings
        assert any(
            document.canonical_url == server + "/public/page" for document in web
        )
        assert store.get_fetch(url_hash(server + "/public/page")) is not None
    finally:
        store.close()


"""Compliant HTTP fetcher for the semantic-scraping frontier.

The fetcher is deliberately conservative:

- only ``http``/``https``;
- robots.txt is fetched and enforced before every request;
- redirects are followed manually and re-checked at every hop;
- private/loopback/link-local addresses are blocked unless explicitly allowed;
- per-domain rate limits and robots ``Crawl-delay`` are respected;
- PDFs, binaries, oversized bodies, and non-text content types are rejected;
- only an extracted text excerpt is cached, never a PDF or full article;
- conditional requests (ETag/Last-Modified) reuse the cache.

It never downloads PDFs and never stores full article bodies.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Sequence
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel, Field

from ...common import utc_now_iso
from ...config import Settings
from .identity import canonicalize_url, content_sha512, url_hash
from .links import LinkCandidate
from .models import Document

_DEFAULT_ACCEPT = (
    "text/html,application/xhtml+xml,application/json;q=0.9,"
    "application/xml;q=0.8,text/plain;q=0.7,*/*;q=0.1"
)
_ALLOWED_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "text/plain",
        "text/csv",
        "application/json",
        "application/ld+json",
        "application/xml",
        "text/xml",
    }
)
_BINARY_PREFIXES = ("image/", "audio/", "video/", "font/")
_BINARY_TYPES = frozenset(
    {
        "application/pdf",
        "application/octet-stream",
        "application/zip",
        "application/gzip",
        "application/x-gzip",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)
_TRUNCATION_MARKER = "\n[truncated]"


class FetchResult(BaseModel):
    url: str
    final_url: str
    status: int
    content_type: str = ""
    title: str | None = None
    text: str = ""
    links: list[str] = Field(default_factory=list)
    content_sha512: str = ""
    from_cache: bool = False


@dataclass(frozen=True)
class RobotsRules:
    allow_all: bool = False
    disallow_all: bool = False
    rules: tuple[tuple[bool, str], ...] = ()
    crawl_delay: float | None = None
    fetched_at: float = 0.0


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _pattern_matches(pattern: str, path: str) -> bool:
    if not pattern:
        return False
    anchor = ""
    if pattern.endswith("$"):
        pattern = pattern[:-1]
        anchor = "$"
    regex = "^" + re.escape(pattern).replace(r"\*", ".*") + anchor
    return re.match(regex, path) is not None


def robots_allowed(rules: RobotsRules, path: str) -> bool:
    if rules.allow_all:
        return True
    if rules.disallow_all:
        return False
    best_length = -1
    best_allow = True
    for allow, pattern in rules.rules:
        if not pattern:
            continue
        if _pattern_matches(pattern, path):
            length = len(pattern)
            if length > best_length or (length == best_length and allow):
                best_length = length
                best_allow = allow
    return best_allow


def parse_robots(text: str, user_agent: str) -> RobotsRules:
    """Parse robots.txt and return the rules for ``user_agent``.

    Empty or unmatched groups allow everything. ``Crawl-delay`` is the maximum
    value across matching groups. Longest-match wins; ties go to ``Allow``.
    """
    groups: list[tuple[tuple[str, ...], tuple[tuple[bool, str], ...], float | None]] = []
    agents: list[str] = []
    rules: list[tuple[bool, str]] = []
    crawl_delay: float | None = None

    def flush() -> None:
        nonlocal agents, rules, crawl_delay
        if agents:
            groups.append((tuple(agents), tuple(rules), crawl_delay))
        agents = []
        rules = []
        crawl_delay = None

    for raw in (text or "").splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key == "user-agent":
            if rules or crawl_delay is not None:
                flush()
            agents.append(value.lower())
        elif key in {"allow", "disallow"}:
            if not agents:
                continue
            rules.append((key == "allow", value))
        elif key == "crawl-delay":
            if not agents:
                continue
            try:
                delay = float(value)
            except ValueError:
                continue
            if delay >= 0:
                crawl_delay = max(crawl_delay or 0.0, delay)
    flush()

    token = user_agent.split("/", 1)[0].strip().lower()
    lowered_agent = user_agent.strip().lower()
    specific = [
        group
        for group in groups
        if token in group[0] or lowered_agent in group[0]
    ]
    matched = specific or [group for group in groups if "*" in group[0]]
    if not matched:
        return RobotsRules(allow_all=True)

    merged: list[tuple[bool, str]] = []
    delay: float | None = None
    for _, group_rules, group_delay in matched:
        merged.extend(group_rules)
        if group_delay is not None:
            delay = max(delay or 0.0, group_delay)
    merged = [(allow, pattern) for allow, pattern in merged if pattern]
    if not merged:
        return RobotsRules(allow_all=True, crawl_delay=delay)
    return RobotsRules(rules=tuple(merged), crawl_delay=delay)


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def _is_private_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_private_host(host: str) -> bool:
    host = (host or "").strip().strip("[]").lower()
    if not host:
        return True
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    if not infos:
        return True
    for info in infos:
        address = info[4][0]
        if _is_private_address(address):
            return True
    return False


class DomainRateLimiter:
    """Per-domain minimum interval, with injectable clock/sleep for tests."""

    def __init__(
        self,
        min_interval: float,
        *,
        clock: Any = time.monotonic,
        sleep: Any = time.sleep,
    ) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, domain: str, min_interval: float | None = None) -> None:
        interval = self.min_interval if min_interval is None else max(
            0.0, float(min_interval)
        )
        if interval <= 0:
            with self._lock:
                self._last[domain] = self._clock()
            return
        with self._lock:
            now = self._clock()
            last = self._last.get(domain)
            if last is not None:
                delay = interval - (now - last)
                if delay > 0:
                    self._sleep(delay)
                    now = self._clock()
            self._last[domain] = now


class RobotsCache:
    """Fetch and cache robots.txt rules per origin."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        user_agent: str,
        timeout: float,
        ttl_seconds: int = 86400,
        clock: Any = time.monotonic,
        warnings: list[str] | None = None,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._timeout = timeout
        self._ttl = ttl_seconds
        self._clock = clock
        self.warnings = warnings if warnings is not None else []
        self._cache: dict[str, RobotsRules] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def rules_for(self, url: str) -> RobotsRules:
        origin = _origin(url)
        with self._lock:
            cached = self._cache.get(origin)
            if cached is not None and self._clock() - cached.fetched_at <= self._ttl:
                return cached
        rules = self._fetch_rules(origin)
        with self._lock:
            self._cache[origin] = rules
        return rules

    def _fetch_rules(self, origin: str) -> RobotsRules:
        robots_url = origin + "/robots.txt"
        try:
            response = self._client.get(
                robots_url,
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
                follow_redirects=False,
            )
            self._client.cookies.clear()
        except httpx.HTTPError as exc:
            self._client.cookies.clear()
            self.warnings.append(
                f"robots_fetch_failed:{origin}:{type(exc).__name__}; treating as disallow"
            )
            return RobotsRules(disallow_all=True, fetched_at=self._clock())
        status = response.status_code
        if 300 <= status < 400:
            self.warnings.append(
                f"robots_redirect:{origin}; treating as disallow"
            )
            return RobotsRules(disallow_all=True, fetched_at=self._clock())
        if status in {401, 403}:
            return RobotsRules(disallow_all=True, fetched_at=self._clock())
        if status in {404, 410} or 400 <= status < 500:
            return RobotsRules(allow_all=True, fetched_at=self._clock())
        if status >= 500:
            self.warnings.append(
                f"robots_status_{status}:{origin}; treating as disallow"
            )
            return RobotsRules(disallow_all=True, fetched_at=self._clock())
        if status < 200 or status >= 300:
            return RobotsRules(disallow_all=True, fetched_at=self._clock())
        parsed = parse_robots(response.text, self._user_agent)
        return RobotsRules(
            allow_all=parsed.allow_all,
            disallow_all=parsed.disallow_all,
            rules=parsed.rules,
            crawl_delay=parsed.crawl_delay,
            fetched_at=self._clock(),
        )


class _HTMLTextExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self.description: str | None = None
        self.nofollow = False
        self.noarchive = False
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {name.lower(): value for name, value in attrs}
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = (attributes.get("name") or "").lower()
            content = (attributes.get("content") or "").lower()
            if name in {"description", "og:description"} and attributes.get("content"):
                self.description = attributes["content"].strip()
            if name == "robots":
                if "nofollow" in content or "none" in content:
                    self.nofollow = True
                if "noarchive" in content:
                    self.noarchive = True
        elif tag == "a" and attributes.get("href"):
            href = attributes["href"].strip()
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                absolute = urljoin(self.base_url, href)
                if urlsplit(absolute).scheme in {"http", "https"}:
                    self.links.append(absolute)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        else:
            self.text_parts.append(text)


def _dedupe_links(links: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for link in links:
        if not link or link in seen:
            continue
        seen.add(link)
        result.append(link)
    return result


def _cap_text(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - len(_TRUNCATION_MARKER))] + _TRUNCATION_MARKER


def _content_type_allowed(content_type: str) -> bool:
    value = (content_type or "").split(";", 1)[0].strip().lower()
    if not value:
        return True
    if value in _BINARY_TYPES or value.startswith(_BINARY_PREFIXES):
        return False
    if value in _ALLOWED_CONTENT_TYPES:
        return True
    return value.startswith("text/")


def _charset(content_type: str) -> str:
    match = re.search(r"charset=([\w.-]+)", content_type or "", re.IGNORECASE)
    return match.group(1) if match else "utf-8"


class Fetcher:
    """Robots-aware, rate-limited, cache-backed public-page fetcher."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: Any,
        client: httpx.Client | None = None,
        warnings: list[str] | None = None,
        allow_private_networks: bool | None = None,
        clock: Any = time.monotonic,
        sleep: Any = time.sleep,
    ) -> None:
        self.settings = settings
        self.store = store
        self.warnings = warnings if warnings is not None else []
        self.allow_private_networks = (
            settings.scrape_allow_private_networks
            if allow_private_networks is None
            else allow_private_networks
        )
        self._clock = clock
        self._owned_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(settings.scrape_fetch_timeout),
            follow_redirects=False,
            headers={"User-Agent": settings.scrape_user_agent},
        )
        self._rate = DomainRateLimiter(
            settings.scrape_rate_limit_seconds, clock=clock, sleep=sleep
        )
        self._robots = RobotsCache(
            self._client,
            user_agent=settings.scrape_user_agent,
            timeout=settings.scrape_fetch_timeout,
            warnings=self.warnings,
            clock=clock,
        )

    def close(self) -> None:
        if self._owned_client:
            self._client.close()

    def _warn(self, message: str) -> None:
        self.warnings.append(message)

    def _host_allowed(self, host: str) -> bool:
        host = (host or "").strip().lower().rstrip(".")
        blocked = self.settings.scrape_blocked_domains
        allowed = self.settings.scrape_allowed_domains
        if blocked and any(host == item or host.endswith("." + item) for item in blocked):
            return False
        if allowed and not any(
            host == item or host.endswith("." + item) for item in allowed
        ):
            return False
        return True

    def _validate_url(self, url: str) -> str | None:
        try:
            canonical = canonicalize_url(url)
        except ValueError as exc:
            self._warn(f"fetch_invalid_url:{url}:{exc}")
            return None
        parts = urlsplit(canonical)
        host = (parts.hostname or "").lower()
        if not host:
            self._warn(f"fetch_invalid_host:{url}")
            return None
        if not self._host_allowed(host):
            self._warn(f"fetch_domain_blocked:{host}")
            return None
        if not self.allow_private_networks and is_private_host(host):
            self._warn(f"fetch_private_network_blocked:{host}")
            return None
        return canonical

    def _read_limited(self, response: httpx.Response) -> bytes | None:
        length = response.headers.get("content-length")
        if length:
            try:
                if int(length) > self.settings.scrape_max_bytes:
                    self._warn(
                        f"fetch_too_large:{response.request.url}:{length}"
                    )
                    return None
            except ValueError:
                pass
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > self.settings.scrape_max_bytes:
                self._warn(f"fetch_too_large:{response.request.url}")
                return None
            chunks.append(chunk)
        return b"".join(chunks)

    def _request(
        self, url: str, headers: dict[str, str]
    ) -> tuple[int, dict[str, str], bytes, str] | None:
        current = url
        for _ in range(self.settings.scrape_max_redirects + 1):
            validated = self._validate_url(current)
            if validated is None:
                return None
            current = validated
            rules = self._robots.rules_for(current)
            path = urlsplit(current).path or "/"
            if not robots_allowed(rules, path):
                self._warn(f"fetch_robots_disallowed:{current}")
                return None
            delay = max(
                self.settings.scrape_rate_limit_seconds,
                rules.crawl_delay or 0.0,
            )
            self._rate.wait(_origin(current), delay)
            try:
                with self._client.stream(
                    "GET", current, headers=headers, follow_redirects=False
                ) as response:
                    status = response.status_code
                    response_headers = {
                        key.lower(): value for key, value in response.headers.items()
                    }
                    if status == 304:
                        return status, response_headers, b"", current
                    if 300 <= status < 400:
                        location = response_headers.get("location")
                        if not location:
                            self._warn(f"fetch_redirect_without_location:{current}")
                            return None
                        current = urljoin(current, location)
                        continue
                    content = b""
                    if status == 200:
                        limited = self._read_limited(response)
                        if limited is None:
                            return None
                        content = limited
                    return status, response_headers, content, current
            except httpx.TimeoutException:
                self._warn(f"fetch_timeout:{current}")
                return None
            except httpx.HTTPError as exc:
                self._warn(f"fetch_failed:{current}:{type(exc).__name__}")
                return None
            finally:
                self._client.cookies.clear()
        self._warn(f"fetch_too_many_redirects:{url}")
        return None

    def _cached(self, key: str, *, require_fresh: bool = True) -> FetchResult | None:
        row = self.store.get_fetch(key)
        if not row:
            return None
        try:
            fetched = datetime.fromisoformat(
                str(row["fetched_at"]).replace("Z", "+00:00")
            )
        except ValueError:
            return None
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - fetched).total_seconds()
        if require_fresh and age > float(row.get("ttl_seconds") or 0):
            return None
        if not (200 <= int(row.get("status") or 0) < 300):
            return None
        try:
            links = json.loads(row.get("links_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            links = []
        return FetchResult(
            url=row["url"],
            final_url=row["url"],
            status=int(row["status"]),
            content_type=row.get("content_type") or "",
            title=row.get("title"),
            text=row.get("body") or "",
            links=[str(item) for item in links if isinstance(item, str)],
            content_sha512=row.get("content_sha512") or "",
            from_cache=True,
        )

    def _conditional_headers(self, key: str) -> dict[str, str]:
        row = self.store.get_fetch(key)
        headers: dict[str, str] = {}
        if row:
            if row.get("etag"):
                headers["If-None-Match"] = str(row["etag"])
            if row.get("last_modified"):
                headers["If-Modified-Since"] = str(row["last_modified"])
        return headers

    def fetch(self, url: str, *, use_cache: bool = True) -> FetchResult | None:
        canonical = self._validate_url(url)
        if canonical is None:
            return None
        key = url_hash(canonical)
        rules = self._robots.rules_for(canonical)
        path = urlsplit(canonical).path or "/"
        if not robots_allowed(rules, path):
            self._warn(f"fetch_robots_disallowed:{canonical}")
            return None
        cached = self._cached(key) if use_cache else None
        if cached is not None:
            return cached

        headers = {
            "User-Agent": self.settings.scrape_user_agent,
            "Accept": _DEFAULT_ACCEPT,
            "Accept-Language": "en;q=0.8",
            "DNT": "1",
        }
        headers.update(self._conditional_headers(key))
        response = self._request(canonical, headers)
        if response is None:
            return None
        status, response_headers, body, final_url = response

        if status == 304:
            cached = self._cached(key, require_fresh=False)
            if cached is None:
                return None
            row = self.store.get_fetch(key) or {}
            self.store.record_fetch(
                url_hash_value=key,
                url=final_url,
                status=int(row.get("status") or 200),
                body=row.get("body") or cached.text,
                etag=row.get("etag"),
                last_modified=row.get("last_modified"),
                ttl_seconds=self.settings.scrape_ttl_seconds,
                title=row.get("title"),
                content_type=row.get("content_type"),
                links=json.loads(row.get("links_json") or "[]"),
                content_sha512=row.get("content_sha512"),
            )
            return cached

        if status != 200:
            self._warn(f"fetch_status_{status}:{final_url}")
            return None

        if urlsplit(final_url).path.lower().endswith(".pdf"):
            self._warn(f"fetch_pdf_rejected:{final_url}")
            return None

        content_type = response_headers.get("content-type", "")
        if not _content_type_allowed(content_type):
            self._warn(f"fetch_content_type_rejected:{content_type}:{final_url}")
            return None

        charset = _charset(content_type)
        try:
            decoded = body.decode(charset, errors="replace")
        except LookupError:
            decoded = body.decode("utf-8", errors="replace")

        base_type = content_type.split(";", 1)[0].strip().lower()
        header_robots = response_headers.get("x-robots-tag", "").lower()
        if base_type in {"text/html", "application/xhtml+xml"}:
            parser = _HTMLTextExtractor(final_url)
            try:
                parser.feed(decoded)
            except Exception:  # noqa: BLE001 - malformed HTML is not fatal
                pass
            title = " ".join(parser.title_parts).strip() or None
            text = _cap_text(
                " ".join(parser.text_parts),
                self.settings.scrape_max_text_chars,
            )
            nofollow = parser.nofollow or "nofollow" in header_robots or "none" in header_robots
            noarchive = parser.noarchive or "noarchive" in header_robots
            links = [] if nofollow else _dedupe_links(parser.links)
            if not title and parser.description:
                title = parser.description
        else:
            title = None
            text = _cap_text(decoded, self.settings.scrape_max_text_chars)
            nofollow = "nofollow" in header_robots or "none" in header_robots
            noarchive = "noarchive" in header_robots
            links = []

        digest = content_sha512(text)
        if not noarchive:
            self.store.record_fetch(
                url_hash_value=key,
                url=final_url,
                status=status,
                body=text,
                etag=response_headers.get("etag"),
                last_modified=response_headers.get("last-modified"),
                ttl_seconds=self.settings.scrape_ttl_seconds,
                title=title,
                content_type=content_type,
                links=links,
                content_sha512=digest,
            )
        return FetchResult(
            url=canonical,
            final_url=final_url,
            status=status,
            content_type=content_type,
            title=title,
            text=text,
            links=links,
            content_sha512=digest,
            from_cache=False,
        )

    def fetch_document(self, candidate: LinkCandidate) -> Document | None:
        result = self.fetch(candidate.url)
        if result is None:
            return None
        canonical = result.final_url
        digest = url_hash(canonical)
        metadata: dict[str, Any] = {
            "public_text": result.text,
            "links": result.links,
            "content_type": result.content_type,
            "status_code": result.status,
            "fetched_at": utc_now_iso(),
            "fetch_cache": "hit" if result.from_cache else "miss",
        }
        if candidate.abstract:
            metadata["abstract"] = candidate.abstract
        if candidate.doi:
            metadata["doi"] = candidate.doi
        return Document(
            id=f"web_{digest}",
            kind="web",
            source="web",
            source_id=canonical,
            title=result.title or candidate.title or canonical,
            summary="",
            summary_embedding=[],
            embedding_model="",
            canonical_url=canonical,
            url_hash=digest,
            content_sha512=result.content_sha512,
            authors=list(candidate.authors),
            publisher=None,
            published_at=str(candidate.year) if candidate.year is not None else None,
            license=None,
            metadata=metadata,
            status="fetched",
        )

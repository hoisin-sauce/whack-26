from __future__ import annotations

import socket
from typing import Any

import httpx
import pytest

from whack26.config import Settings
from whack26.research.scrape.fetch import Fetcher, is_private_host
from whack26.store import FieldStore

PUBLIC_IP = "93.184.216.34"
PUBLIC_V6 = "2606:4700:4700::1111"


def addrinfo_for(*addresses: str) -> list[Any]:
    """Build getaddrinfo-style tuples for literal addresses."""
    infos: list[Any] = []
    for address in addresses:
        if ":" in address:
            infos.append(
                (
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    (address, 0, 0, 0),
                )
            )
        else:
            infos.append(
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    (address, 0),
                )
            )
    return infos


def patch_resolution(
    monkeypatch: pytest.MonkeyPatch, mapping: dict[str, tuple[str, ...]]
) -> None:
    def fake_getaddrinfo(host: str, port: object, *args: object, **kwargs: object):
        key = str(host).strip("[]").lower()
        if key not in mapping:
            raise socket.gaierror(socket.EAI_NONAME, "name or service not known")
        return addrinfo_for(*mapping[key])

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "scrape_fetch_enabled": True,
        "scrape_allow_private_networks": False,
        "scrape_rate_limit_seconds": 0.0,
        "scrape_fetch_timeout": 1.0,
        "scrape_max_redirects": 2,
    }
    values.update(overrides)
    return Settings.from_env({}, mock=True).with_overrides(**values)


@pytest.mark.parametrize(
    "host",
    [
        "::1",  # IPv6 loopback
        "fc00::1",  # IPv6 unique-local
        "fe80::1",  # IPv6 link-local
        "127.0.0.1",  # IPv4 loopback
        "10.0.0.5",  # IPv4 private
        "169.254.1.1",  # IPv4 link-local
    ],
    ids=[
        "ipv6-loopback",
        "ipv6-unique-local",
        "ipv6-link-local",
        "ipv4-loopback",
        "ipv4-private",
        "ipv4-link-local",
    ],
)
def test_is_private_host_blocks_loopback_private_and_link_local(host: str) -> None:
    assert is_private_host(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "240.0.0.1",  # IPv4 reserved
        "224.0.0.1",  # IPv4 multicast
        "0.0.0.0",  # IPv4 unspecified
        "::",  # IPv6 unspecified
        "ff02::1",  # IPv6 multicast
    ],
    ids=[
        "reserved",
        "multicast",
        "unspecified-ipv4",
        "unspecified-ipv6",
        "multicast-ipv6",
    ],
)
def test_is_private_host_blocks_reserved_multicast_and_unspecified(
    host: str,
) -> None:
    assert is_private_host(host) is True


@pytest.mark.parametrize("host", [PUBLIC_IP, PUBLIC_V6, "8.8.8.8"])
def test_is_private_host_allows_public_addresses(host: str) -> None:
    assert is_private_host(host) is False


def test_is_private_host_blocks_hostname_resolving_to_private_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_resolution(
        monkeypatch,
        {
            "private.example.test": ("127.0.0.1",),
            "mixed.example.test": (PUBLIC_IP, "10.1.2.3"),
        },
    )

    assert is_private_host("private.example.test") is True
    # A single private answer is enough to block a multi-address hostname.
    assert is_private_host("mixed.example.test") is True


def test_is_private_host_fails_closed_on_dns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_getaddrinfo(
        host: str, port: object, *args: object, **kwargs: object
    ) -> list[Any]:
        raise socket.gaierror(socket.EAI_NONAME, "name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", failing_getaddrinfo)

    assert is_private_host("unresolvable.example.test") is True


def test_is_private_host_fails_closed_on_empty_dns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [])

    assert is_private_host("empty.example.test") is True


def test_is_private_host_allows_hostname_resolving_to_public_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_resolution(
        monkeypatch,
        {"public.example.test": (PUBLIC_IP,)},
    )

    assert is_private_host("public.example.test") is False


def test_fetch_rejects_a_userinfo_url_to_a_private_host() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        raise AssertionError("a blocked URL must not be requested")

    store = FieldStore(":memory:", dims=2)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = Fetcher(
        settings=make_settings(),
        store=store,
        client=client,
        allow_private_networks=False,
    )
    try:
        assert fetcher.fetch("https://user:pass@127.0.0.1/secret") is None
        assert calls == []
        assert any(
            "fetch_private_network_blocked:127.0.0.1" in warning
            for warning in fetcher.warnings
        )
    finally:
        fetcher.close()
        client.close()
        store.close()


def test_fetch_rejects_redirect_to_private_host_before_requesting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_resolution(
        monkeypatch,
        {
            "public.example.test": (PUBLIC_IP,),
            "private.example.test": ("127.0.0.1",),
        },
    )
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host == "public.example.test" and request.url.path == "/robots.txt":
            return httpx.Response(404, text="")
        if str(request.url) == "https://public.example.test/page":
            return httpx.Response(
                302,
                headers={"Location": "http://private.example.test/secret"},
            )
        return httpx.Response(500, text="unexpected request")

    store = FieldStore(":memory:", dims=2)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = Fetcher(
        settings=make_settings(),
        store=store,
        client=client,
        allow_private_networks=False,
    )
    try:
        result = fetcher.fetch("https://public.example.test/page")
    finally:
        fetcher.close()
        client.close()
        store.close()

    assert result is None
    assert requested == [
        "https://public.example.test/robots.txt",
        "https://public.example.test/page",
    ]
    assert "http://private.example.test/secret" not in requested
    assert not any("private.example.test" in url for url in requested)
    assert any(
        warning == "fetch_private_network_blocked:private.example.test"
        for warning in fetcher.warnings
    )

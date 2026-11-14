"""Outbound HTTP with an SSRF guard, size cap and timeout.

Every URL we download comes from a feed or a web page, i.e. from strangers.
Before each request (and each redirect hop) the host is resolved and the
request is refused unless every address is a public one, so a malicious feed
cannot make the Pi fetch http://localhost:..., the router, or other machines
on the LAN / tailnet.

Known limit: the check and the actual connection resolve DNS separately, so a
DNS-rebinding attacker could in theory slip through. That is acceptable for a
single-user reader; the fetched bytes are only ever parsed as text or images.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import requests

from . import config

MAX_REDIRECTS = 5


class UnsafeURLError(ValueError):
    """The URL points somewhere we refuse to fetch."""


class TooLargeError(ValueError):
    """The response body exceeded the size cap."""


@dataclass
class Response:
    status: int
    url: str
    headers: dict[str, str]
    content: bytes


def is_public_ip(address: str) -> bool:
    """True only for globally routable addresses."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    # is_global is False for private, loopback, link-local, CGNAT (tailnet
    # 100.64/10), multicast, reserved and unspecified ranges.
    return ip.is_global and not ip.is_multicast


def resolve_host(host: str) -> list[str]:
    """All IP addresses a hostname resolves to."""
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


def assert_safe_url(url: str) -> None:
    """Raise UnsafeURLError unless url is http(s) and resolves only to public IPs."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError(f"scheme not allowed: {parts.scheme or '(none)'}")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("no host in URL")
    try:
        addresses = resolve_host(host)
    except OSError as exc:
        raise UnsafeURLError(f"cannot resolve {host}: {exc}") from exc
    if not addresses:
        raise UnsafeURLError(f"cannot resolve {host}")
    for address in addresses:
        if not is_public_ip(address):
            raise UnsafeURLError(f"{host} resolves to non-public address {address}")


def safe_get(
    url: str,
    *,
    max_bytes: int,
    headers: dict[str, str] | None = None,
    timeout: float = config.HTTP_TIMEOUT,
) -> Response:
    """GET a URL, following redirects manually so each hop is re-checked.

    Raises UnsafeURLError, TooLargeError or requests.RequestException.
    """
    request_headers = {"User-Agent": config.USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    request_headers.update(headers or {})
    for _ in range(MAX_REDIRECTS + 1):
        assert_safe_url(url)
        with requests.get(
            url, headers=request_headers, timeout=timeout, stream=True, allow_redirects=False
        ) as resp:
            if resp.status_code in (301, 302, 303, 307, 308) and "location" in resp.headers:
                url = urljoin(url, resp.headers["location"])
                continue
            declared = resp.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > max_bytes:
                raise TooLargeError(f"{url}: {declared} bytes > cap {max_bytes}")
            body = bytearray()
            for chunk in resp.iter_content(chunk_size=65536):
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise TooLargeError(f"{url}: body > cap {max_bytes}")
            return Response(resp.status_code, url, {k.lower(): v for k, v in resp.headers.items()}, bytes(body))
    raise UnsafeURLError(f"too many redirects for {url}")

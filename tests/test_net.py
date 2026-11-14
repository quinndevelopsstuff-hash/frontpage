import pytest

from core import net


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.3.4",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata / link-local
        "100.100.100.100",  # tailnet (CGNAT) range
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fd00::1",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "224.0.0.1",
        "not-an-ip",
    ],
)
def test_non_public_addresses_are_rejected(address):
    assert not net.is_public_ip(address)


@pytest.mark.parametrize("address", ["1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"])
def test_public_addresses_are_allowed(address):
    assert net.is_public_ip(address)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com", "//example.com", ""])
def test_only_http_and_https(url):
    with pytest.raises(net.UnsafeURLError):
        net.assert_safe_url(url)


def test_hostname_resolving_to_private_ip_is_blocked(monkeypatch):
    monkeypatch.setattr(net, "resolve_host", lambda host: ["192.168.1.10"])
    with pytest.raises(net.UnsafeURLError):
        net.assert_safe_url("https://innocent.example.com/image.jpg")


def test_any_private_address_in_the_answer_blocks(monkeypatch):
    monkeypatch.setattr(net, "resolve_host", lambda host: ["93.184.216.34", "127.0.0.1"])
    with pytest.raises(net.UnsafeURLError):
        net.assert_safe_url("https://example.com/")


def test_literal_loopback_and_localhost_blocked():
    for url in ("http://127.0.0.1:9092/", "http://localhost/", "http://[::1]/"):
        with pytest.raises(net.UnsafeURLError):
            net.assert_safe_url(url)


def test_public_host_passes(monkeypatch):
    monkeypatch.setattr(net, "resolve_host", lambda host: ["93.184.216.34"])
    net.assert_safe_url("https://example.com/feed.xml")


class _FakeResponse:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code, self.headers, self._body = status, headers or {}, body

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_redirect_to_private_address_is_blocked(monkeypatch):
    monkeypatch.setattr(net, "resolve_host", lambda host: ["127.0.0.1"] if host == "localhost" else ["93.184.216.34"])
    monkeypatch.setattr(
        net.requests, "get", lambda url, **kw: _FakeResponse(302, {"location": "http://localhost:9092/sources"})
    )
    with pytest.raises(net.UnsafeURLError):
        net.safe_get("https://example.com/redirect", max_bytes=1000)


def test_size_cap_is_enforced(monkeypatch):
    monkeypatch.setattr(net, "resolve_host", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(net.requests, "get", lambda url, **kw: _FakeResponse(200, {}, b"x" * 5000))
    with pytest.raises(net.TooLargeError):
        net.safe_get("https://example.com/big", max_bytes=1000)
    assert net.safe_get("https://example.com/big", max_bytes=10_000).content == b"x" * 5000

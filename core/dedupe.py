"""URL canonicalisation and cheap near-duplicate title detection."""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query parameters that only carry tracking information.
_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref_src", "cmpid", "ncid"}
_STOPWORDS = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "at", "by", "with"}
TITLE_SIMILARITY_THRESHOLD = 0.8


def canonicalize_url(url: str) -> str:
    """Normalise a URL so the same article always maps to the same string.

    Lower-cases scheme and host, drops fragments, default ports, utm_* and
    other tracking parameters, and a trailing slash on the path.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, host, path, urlencode(query), ""))


def title_tokens(title: str) -> frozenset[str]:
    """Lower-cased significant words of a title."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def title_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets (0..1)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class TitleIndex:
    """Remembers recent titles and answers "have I seen nearly this one?".

    A linear scan over a few thousand small sets is fast enough and needs no
    extra dependency.
    """

    def __init__(self, titles: list[str] | None = None, threshold: float = TITLE_SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self._seen: list[frozenset[str]] = []
        for title in titles or []:
            self.add(title)

    def add(self, title: str) -> None:
        tokens = title_tokens(title)
        if tokens:
            self._seen.append(tokens)

    def is_duplicate(self, title: str) -> bool:
        tokens = title_tokens(title)
        # Very short titles ("Open thread") collide by accident; only exact matches count.
        if len(tokens) < 4:
            return tokens in self._seen if tokens else False
        return any(title_similarity(tokens, other) >= self.threshold for other in self._seen)

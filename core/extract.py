"""Article text, og:image extraction and image caching."""
from __future__ import annotations

import hashlib
import html
import io
import logging
import re
import threading
from urllib.parse import urljoin

from . import config, net

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_META_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""([a-zA-Z:_-]+)\s*=\s*("([^"]*)"|'([^']*)')""")
_MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs


def html_to_text(fragment: str) -> str:
    """Strip tags and collapse whitespace (for RSS summaries)."""
    text = _TAG_RE.sub(" ", fragment or "")
    return _SPACE_RE.sub(" ", html.unescape(text)).strip()


def truncate(text: str, limit: int) -> str:
    """Cut text at a word boundary, adding an ellipsis when shortened."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def extract_text(page_html: str) -> str | None:
    """Main article text via trafilatura, capped; None when extraction fails."""
    try:
        import trafilatura

        text = trafilatura.extract(page_html, include_comments=False, include_tables=False)
    except Exception as exc:  # trafilatura can raise on odd markup; never fatal
        log.debug("extraction failed: %s", exc)
        return None
    if not text or len(text) < 200:
        return None
    return text[: config.MAX_TEXT_CHARS]


def find_og_image(page_html: str, base_url: str) -> str | None:
    """The page's og:image (or twitter:image) as an absolute URL."""
    found: dict[str, str] = {}
    for tag in _META_RE.findall(page_html[:200_000]):
        attrs = {m.group(1).lower(): (m.group(3) or m.group(4) or "") for m in _ATTR_RE.finditer(tag)}
        key = (attrs.get("property") or attrs.get("name") or "").lower()
        if key in ("og:image", "og:image:url", "og:image:secure_url", "twitter:image") and attrs.get("content"):
            found.setdefault(key, html.unescape(attrs["content"]))
    for key in ("og:image", "og:image:secure_url", "og:image:url", "twitter:image"):
        if key in found:
            return urljoin(base_url, found[key].strip())
    return None


def fetch_page(url: str, total_timeout: float | None = None) -> tuple[str, str] | None:
    """Download an article page. Returns (html, final_url) or None, quietly."""
    try:
        resp = net.safe_get(url, max_bytes=config.MAX_PAGE_BYTES, total_timeout=total_timeout)
    except Exception as exc:
        log.debug("page fetch failed for %s: %s", url, exc)
        return None
    if resp.status != 200 or "html" not in resp.headers.get("content-type", "html"):
        return None
    return resp.content.decode("utf-8", errors="replace"), resp.url


def cache_image(
    url: str, min_size: tuple[int, int] = (200, 100), total_timeout: float | None = None
) -> str | None:
    """Download, shrink and store an image. Returns the cached file name or None.

    The file name is a hash of the URL, so re-fetching the same image is free.
    Images smaller than min_size (width, height) are rejected as icons/pixels.
    """
    from PIL import Image

    name = hashlib.sha256(url.encode()).hexdigest()[:24] + ".webp"
    target = config.images_dir() / name
    if target.exists():
        return name
    try:
        resp = net.safe_get(url, max_bytes=config.MAX_IMAGE_BYTES, total_timeout=total_timeout)
        if resp.status != 200:
            return None
        Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
        with Image.open(io.BytesIO(resp.content)) as img:
            img.load()
            if img.width < min_size[0] or img.height < min_size[1]:  # tracking pixels, icons
                return None
            img = img.convert("RGB")
            if img.width > config.IMAGE_WIDTH:
                height = round(img.height * config.IMAGE_WIDTH / img.width)
                img = img.resize((config.IMAGE_WIDTH, height), Image.LANCZOS)
            partial = target.with_suffix(f".{threading.get_ident()}.part")
            img.save(partial, "WEBP", quality=80)
            partial.replace(target)
    except Exception as exc:
        log.debug("image cache failed for %s: %s", url, exc)
        target.unlink(missing_ok=True)
        return None
    return name

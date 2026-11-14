"""Reader view: article HTML extraction, sanitising and image localisation.

Lifecycle of an article's reader content (articles.reader_state):

    NULL      not tried yet. Opening the article downloads the page and runs
              trafilatura in HTML mode ("lazy backfill").
    'remote'  sanitised HTML is stored, but its images still point at the
              original site. The fetch job stores this state when it already
              had the page (or full feed content) in hand, so it never
              downloads inline images for articles nobody opens.
    'ready'   images have been run through the image cache (data/images/),
              so the HTML is final. Set on first open.
    'failed'  extraction failed or gave under READER_MIN_CHARS of text; the
              reader shows the summary and an "Open original" button. Retried
              on open once READER_RETRY_SECONDS have passed.

Everything that reaches the browser goes through sanitize_html, an allowlist
of tags and attributes (nh3). Stored HTML is sanitised again on every render,
so a sanitiser fix also covers what is already in the database.
"""
from __future__ import annotations

import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urljoin, urlsplit

import nh3

from . import config, db, extract
from .dedupe import title_similarity, title_tokens

READER_MIN_CHARS = 300  # less text than this counts as a failed extraction
READER_RETRY_SECONDS = 6 * 3600
MAX_CONTENT_HTML_CHARS = 150_000
PAGE_TOTAL_TIMEOUT = 12.0  # seconds for the lazy page download on open
MAX_INLINE_IMAGES = 12  # images cached per article; the rest stay remote
IMAGE_WORKERS = 4
IMAGE_BUDGET_SECONDS = 8.0  # wall-clock budget for caching one article's images
MIN_INLINE_IMAGE = (60, 40)  # smaller images are icons or tracking pixels

ALLOWED_TAGS = {
    "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "blockquote", "pre", "code",
    "a", "img", "figure", "figcaption",
    "strong", "b", "em", "i", "u", "s", "sub", "sup", "small", "mark",
}  # fmt: skip
# Removed together with everything inside them (not just the tags).
DROP_WITH_CONTENT = {
    "script", "style", "noscript", "template", "iframe", "object", "embed",
    "svg", "math", "form", "button", "select", "textarea", "canvas", "video", "audio",
}  # fmt: skip
ALLOWED_ATTRIBUTES = {"a": {"href", "title"}, "img": {"src", "alt", "title"}, "ol": {"start"}}
FORCED_ATTRIBUTES = {
    "a": {"target": "_blank"},
    "img": {"loading": "lazy", "decoding": "async", "referrerpolicy": "no-referrer"},
}
LOCAL_IMAGE_RE = re.compile(r"^/img/[0-9a-f]{24}\.webp$")
_IMG_WITHOUT_SRC_RE = re.compile(r"<img(?![^>]*\ssrc=)[^>]*>")
_LEADING_HEADING_RE = re.compile(r"^\s*<h[12]>(.*?)</h[12]>", re.DOTALL)
_EMPTY_FIGURE_RE = re.compile(r"<figure>\s*(<figcaption>.*?</figcaption>)?\s*</figure>", re.DOTALL)


# --- sanitising -------------------------------------------------------------


def _url_filter(base_url: str, image_map: dict[str, str] | None) -> Callable[[str, str, str], str | None]:
    """nh3 attribute filter: make URLs absolute and swap in cached images.

    nh3 has already removed javascript:, data: and other non-http(s) schemes
    by the time this runs. Relative URLs are resolved against the article's
    own URL, so nothing in an article can point at Front Page itself, except
    the /img/<hash>.webp paths that this module writes.
    """

    def keep(tag: str, attr: str, value: str) -> str | None:
        if attr not in ("href", "src"):
            return value
        value = value.strip()
        if tag == "img" and attr == "src" and LOCAL_IMAGE_RE.match(value):
            return value
        absolute = urljoin(base_url, value) if base_url else value
        allowed = ("http", "https", "mailto") if attr == "href" else ("http", "https")
        if urlsplit(absolute).scheme not in allowed:
            return None
        if tag == "img" and image_map is not None:
            return image_map.get(absolute, absolute) or None  # "" means drop the image
        return absolute

    return keep


def sanitize_html(html: str, base_url: str, image_map: dict[str, str] | None = None) -> str:
    """Allowlist-sanitise an HTML fragment for the reader.

    Keeps paragraphs, headings, lists, quotes, code, links and images. Drops
    scripts, iframes, forms, styles, classes, ids and every event handler.
    Links open in a new tab without a referrer; images load lazily without
    one. image_map rewrites image URLs (original absolute URL -> local path).
    """
    cleaned = nh3.clean(
        html or "",
        tags=ALLOWED_TAGS,
        clean_content_tags=DROP_WITH_CONTENT,
        attributes=ALLOWED_ATTRIBUTES,
        attribute_filter=_url_filter(base_url, image_map),
        url_schemes={"http", "https", "mailto"},
        link_rel="noopener noreferrer nofollow",
        set_tag_attribute_values=FORCED_ATTRIBUTES,
        strip_comments=True,
    )
    cleaned = _IMG_WITHOUT_SRC_RE.sub("", cleaned)  # images whose URL was rejected
    return _EMPTY_FIGURE_RE.sub("", cleaned).strip()


def text_length(html: str) -> int:
    return len(extract.html_to_text(html))


def _drop_leading_title(html: str, title: str) -> str:
    """Remove a first heading that just repeats the article title."""
    match = _LEADING_HEADING_RE.match(html)
    if match and title_similarity(title_tokens(extract.html_to_text(match.group(1))), title_tokens(title)) >= 0.8:
        return html[match.end() :].lstrip()
    return html


def prepare_html(raw_html: str, base_url: str, title: str) -> str | None:
    """Sanitise article HTML and check there is enough of it; None means "failed"."""
    html = _drop_leading_title(sanitize_html(raw_html, base_url), title)
    if text_length(html) < READER_MIN_CHARS or len(html) > MAX_CONTENT_HTML_CHARS:
        return None
    return html


def extract_html(page_html: str, url: str, title: str) -> str | None:
    """Main content of a page as sanitised HTML via trafilatura, or None."""
    try:
        import trafilatura

        raw = trafilatura.extract(
            page_html,
            url=url,
            output_format="html",
            include_links=True,
            include_images=True,
            include_formatting=True,
            include_comments=False,
            include_tables=False,
        )
    except Exception:  # odd markup can make trafilatura raise; never fatal
        return None
    if not raw:
        return None
    body = re.search(r"<body[^>]*>(.*)</body>", raw, re.DOTALL)
    return prepare_html(body.group(1) if body else raw, url, title)


# --- images -----------------------------------------------------------------


def image_urls(html: str) -> list[str]:
    """Remote image URLs in sanitised HTML, in document order, unique."""
    seen: list[str] = []
    for url in re.findall(r'<img[^>]*\ssrc="(https?://[^"]+)"', html):
        url = url.replace("&amp;", "&")
        if url not in seen:
            seen.append(url)
    return seen


def localize_images(html: str, base_url: str, hero_url: str | None = None) -> str:
    """Route inline images through the image cache.

    Up to MAX_INLINE_IMAGES are downloaded in parallel within a time budget.
      cached               -> src becomes /img/<hash>.webp
      tried and failed     -> image removed (blocked host, not an image,
                              too small to be content, too large)
      not tried (cap or    -> left pointing at the original site
      out of time)
    An inline copy of the hero image is removed, since the page shows it above.
    """
    urls = image_urls(html)
    image_map: dict[str, str] = {}
    if hero_url:
        image_map[hero_url] = ""  # empty src -> the <img> is dropped
    todo = [u for u in urls if u != hero_url][:MAX_INLINE_IMAGES]
    if todo:
        pool = ThreadPoolExecutor(max_workers=IMAGE_WORKERS)
        futures = {pool.submit(_cache_inline, url): url for url in todo}
        done, _ = wait(futures, timeout=IMAGE_BUDGET_SECONDS)
        pool.shutdown(wait=False, cancel_futures=True)
        for future in done:
            name = future.result()
            image_map[futures[future]] = f"/img/{name}" if name else ""
    return sanitize_html(html, base_url, image_map)


def _cache_inline(url: str) -> str | None:
    try:
        return extract.cache_image(url, min_size=MIN_INLINE_IMAGE, total_timeout=IMAGE_BUDGET_SECONDS)
    except Exception:
        return None


# --- lazy backfill ----------------------------------------------------------


@dataclass
class ReaderContent:
    html: str | None  # None -> show the summary fallback
    state: str


def ensure_reader_content(conn: sqlite3.Connection, article: sqlite3.Row) -> ReaderContent:
    """Make sure an article has reader HTML, extracting it on first open.

    article needs: id, url, title, image_url, content_html, reader_state,
    reader_checked_at. Writes the outcome back to the database.
    """
    now = db.now()
    state, html = article["reader_state"], article["content_html"]
    retry_due = state == "failed" and now - (article["reader_checked_at"] or 0) > READER_RETRY_SECONDS
    if state is None or retry_due:
        page = extract.fetch_page(article["url"], total_timeout=PAGE_TOTAL_TIMEOUT)
        html = extract_html(page[0], page[1], article["title"]) if page else None
        state = "remote" if html else "failed"
        _store(conn, article["id"], html, state, now)
    if state == "remote" and html:
        html = localize_images(html, article["url"], hero_url=article["image_url"])
        state = "ready" if text_length(html) >= READER_MIN_CHARS else "failed"
        _store(conn, article["id"], html if state == "ready" else None, state, now)
    return ReaderContent(html if state == "ready" else None, state)


def _store(conn: sqlite3.Connection, article_id: int, html: str | None, state: str, now: float) -> None:
    conn.execute(
        "UPDATE articles SET content_html = ?, reader_state = ?, reader_checked_at = ? WHERE id = ?",
        (html, state, now, article_id),
    )
    conn.commit()


# --- disk housekeeping ------------------------------------------------------


def sweep_images(conn: sqlite3.Connection, min_age_seconds: float = 86400) -> int:
    """Delete cached images no article refers to any more.

    Referenced = every hero image_path plus the /img/ paths inside reader HTML
    (only 'ready' articles contain local paths). Files younger than a day are
    left alone so a download in progress is never swept.
    """
    referenced = {row[0] for row in conn.execute("SELECT image_path FROM articles WHERE image_path IS NOT NULL")}
    for (html,) in conn.execute("SELECT content_html FROM articles WHERE reader_state = 'ready'"):
        referenced.update(re.findall(r'src="/img/([0-9a-f]{24}\.webp)"', html or ""))
    cutoff = time.time() - min_age_seconds
    removed = 0
    for path in config.images_dir().glob("*.webp"):
        if path.name not in referenced and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed

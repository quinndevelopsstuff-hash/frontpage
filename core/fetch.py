"""The fetch job: `python -m core.fetch` (run by frontpage-fetch.timer).

One run:  sync feeds from feeds.yaml -> download each feed (conditional GET)
-> dedupe -> extract text + image for new articles -> embed -> prior/topic ->
score -> delete old articles.

It is deliberately gentle on the Pi: feeds and pages are fetched one at a
time with a short pause, the number of page downloads per run is capped, and
the systemd unit runs it at low CPU/IO priority.

`python -m core.fetch --check` only tests that every feed URL in feeds.yaml
answers with a parseable feed; it stores nothing.
"""
from __future__ import annotations

import argparse
import calendar
import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass

import feedparser

from . import config, db, embed, extract, net, rank
from .dedupe import TitleIndex, canonicalize_url

log = logging.getLogger("frontpage.fetch")

_ARXIV_PREFIX_RE = re.compile(r"^arXiv:\S+\s+Announce Type:\s*\S+\s+Abstract:\s*", re.IGNORECASE)
_FULL_TEXT_CHARS = 1500  # feed content this long counts as the full article


@dataclass
class Candidate:
    """A feed entry that might become an article."""

    url: str
    title: str
    author: str | None
    published_at: float
    summary: str
    feed_text: str
    image_url: str | None


# --- feeds table ------------------------------------------------------------


def sync_feeds(conn: sqlite3.Connection, seeds: list[config.FeedSeed] | None = None) -> None:
    """Apply feeds.yaml to the feeds table (the yaml file is only a seed).

    New URLs are inserted; for known URLs the descriptive fields are updated.
    muted/removed flags set on the Sources page are never touched here.
    """
    seeds = config.load_feed_seeds() if seeds is None else seeds
    for seed in seeds:
        conn.execute(
            "INSERT INTO feeds (url, name, default_topic, weight, extract, max_items, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(url) DO UPDATE SET name = excluded.name, default_topic = excluded.default_topic, "
            "weight = excluded.weight, extract = excluded.extract, max_items = excluded.max_items",
            (seed.url, seed.name, seed.default_topic, seed.weight, int(seed.extract), seed.max_items, db.now()),
        )
    conn.commit()


def download_feed(url: str, etag: str | None = None, last_modified: str | None = None) -> net.Response:
    """Conditional GET of a feed."""
    headers = {"Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.5"}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    return net.safe_get(url, max_bytes=config.MAX_FEED_BYTES, headers=headers)


def parse_feed(content: bytes) -> list[feedparser.FeedParserDict]:
    """Entries of a feed document; raises ValueError if it is not a feed."""
    parsed = feedparser.parse(content)
    # feedparser is forgiving: an HTML error page "parses" too, just with no
    # detected feed format (version) and no entries.
    if not parsed.entries and not parsed.get("version"):
        raise ValueError(f"not a valid feed: {parsed.get('bozo_exception', 'no feed format detected')}")
    return list(parsed.entries)


def fetch_feed(conn: sqlite3.Connection, feed: sqlite3.Row) -> list[feedparser.FeedParserDict]:
    """Download one feed and record the outcome. Never raises."""
    now = db.now()
    try:
        resp = download_feed(feed["url"], feed["etag"], feed["last_modified"])
        if resp.status == 304:
            entries: list[feedparser.FeedParserDict] = []
        elif resp.status != 200:
            raise ValueError(f"HTTP {resp.status}")
        else:
            entries = parse_feed(resp.content)
        conn.execute(
            "UPDATE feeds SET etag = ?, last_modified = ?, last_fetch_at = ?, last_ok_at = ?, last_error = NULL "
            "WHERE id = ?",
            (
                resp.headers.get("etag") or feed["etag"],
                resp.headers.get("last-modified") or feed["last_modified"],
                now,
                now,
                feed["id"],
            ),
        )
        conn.commit()
        return entries
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:300]
        log.warning("feed failed: %s (%s)", feed["name"], message)
        conn.execute("UPDATE feeds SET last_fetch_at = ?, last_error = ? WHERE id = ?", (now, message, feed["id"]))
        conn.commit()
        return []


# --- entries -> candidates --------------------------------------------------


def entry_image(entry: feedparser.FeedParserDict) -> str | None:
    """The feed's own image for an entry: media thumbnail, media content or enclosure."""
    for thumb in entry.get("media_thumbnail") or []:
        if thumb.get("url"):
            return thumb["url"]
    for media in entry.get("media_content") or []:
        kind = media.get("type", "")
        if media.get("url") and (media.get("medium") == "image" or kind.startswith("image/") or not kind):
            return media["url"]
    for link in entry.get("enclosures") or []:
        if link.get("href") and link.get("type", "").startswith("image/"):
            return link["href"]
    return None


def entry_published(entry: feedparser.FeedParserDict, now: float) -> float:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(key)
        if parsed:
            return min(float(calendar.timegm(parsed)), now)
    return now


def entry_to_candidate(entry: feedparser.FeedParserDict, now: float) -> Candidate | None:
    link, title = entry.get("link"), extract.html_to_text(entry.get("title", ""))
    if not link or not title or not link.startswith(("http://", "https://")):
        return None
    summary = extract.html_to_text(entry.get("summary", ""))
    summary = _ARXIV_PREFIX_RE.sub("", summary)
    content = entry.get("content") or []
    feed_text = extract.html_to_text(content[0].get("value", "")) if content else ""
    return Candidate(
        url=canonicalize_url(link),
        title=title[:300],
        author=(entry.get("author") or None),
        published_at=entry_published(entry, now),
        summary=summary,
        feed_text=feed_text if len(feed_text) > len(summary) else summary,
        image_url=entry_image(entry),
    )


def pick_summary(rss_summary: str, text: str) -> str:
    """A card summary: the RSS one, unless it is empty or link boilerplate."""
    boilerplate = len(rss_summary) < 60 or rss_summary.startswith(("Article URL:", "Comments URL:"))
    source = text if boilerplate and text else rss_summary
    return extract.truncate(source, config.MAX_SUMMARY_CHARS)


# --- storing ----------------------------------------------------------------


class PageBudget:
    """Caps article-page downloads per run and paces them."""

    def __init__(self, limit: int = config.MAX_PAGE_FETCHES_PER_RUN, pause: float = config.PAGE_FETCH_PAUSE):
        self.remaining, self.pause = limit, pause

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        time.sleep(self.pause)
        return True


def enrich(candidate: Candidate, extract_pages: bool, budget: PageBudget) -> tuple[str, str | None]:
    """Full text and image URL for a candidate, downloading the page if useful."""
    text, image_url = candidate.feed_text, candidate.image_url
    wants_page = extract_pages and (len(text) < _FULL_TEXT_CHARS or not image_url)
    if wants_page and budget.take():
        page = extract.fetch_page(candidate.url)
        if page:
            page_html, final_url = page
            if len(text) < _FULL_TEXT_CHARS:
                text = extract.extract_text(page_html) or text
            image_url = image_url or extract.find_og_image(page_html, final_url)
    return text[: config.MAX_TEXT_CHARS], image_url


def store_new_articles(
    conn: sqlite3.Connection,
    feed: sqlite3.Row,
    entries: list[feedparser.FeedParserDict],
    titles: TitleIndex,
    budget: PageBudget,
) -> int:
    """Insert the entries of one feed that we have not seen before."""
    now = db.now()
    cutoff = now - config.RETENTION_DAYS * 86400
    candidates = [c for c in (entry_to_candidate(e, now) for e in entries) if c and c.published_at >= cutoff]
    candidates.sort(key=lambda c: c.published_at, reverse=True)
    added = 0
    for candidate in candidates[: feed["max_items"]]:
        if conn.execute("SELECT 1 FROM articles WHERE url = ?", (candidate.url,)).fetchone():
            continue
        if titles.is_duplicate(candidate.title):
            continue
        text, image_url = enrich(candidate, bool(feed["extract"]), budget)
        image_path = extract.cache_image(image_url) if image_url else None
        conn.execute(
            "INSERT OR IGNORE INTO articles (feed_id, url, title, author, published_at, fetched_at, summary, "
            "content_text, image_url, image_path, topic) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                feed["id"],
                candidate.url,
                candidate.title,
                candidate.author,
                candidate.published_at,
                now,
                pick_summary(candidate.summary, text),
                text,
                image_url,
                image_path,
                feed["default_topic"],
            ),
        )
        conn.commit()
        titles.add(candidate.title)
        added += 1
    return added


def apply_retention(conn: sqlite3.Connection, now: float | None = None) -> int:
    """Delete articles older than 30 days unless saved or given feedback."""
    cutoff = (db.now() if now is None else now) - config.RETENTION_DAYS * 86400
    rows = conn.execute(
        "SELECT id, image_path FROM articles WHERE published_at < ? "
        "AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.article_id = articles.id)",
        (cutoff,),
    ).fetchall()
    conn.executemany("DELETE FROM articles WHERE id = ?", [(r["id"],) for r in rows])
    conn.commit()
    for row in rows:
        still_used = row["image_path"] and conn.execute(
            "SELECT 1 FROM articles WHERE image_path = ?", (row["image_path"],)
        ).fetchone()
        if row["image_path"] and not still_used:
            (config.images_dir() / row["image_path"]).unlink(missing_ok=True)
    return len(rows)


def index_and_score(conn: sqlite3.Connection, embedder: embed.Embedder) -> int:
    """Embed new articles, compute priors/topics, then rescore everything."""
    embed.ensure_backend(conn, embedder)
    embedded = embed.embed_pending(conn, embedder)
    interests, changed = rank.load_interests(conn, embedder)
    rank.update_priors(conn, interests, recompute_all=changed)
    rank.retrain(conn)  # also rescores, so recency decay stays current
    return len(embedded)


# --- entry points -----------------------------------------------------------


def run() -> int:
    """One full fetch run. Returns the number of new articles."""
    conn = db.connect()
    sync_feeds(conn)
    recent = conn.execute("SELECT title FROM articles ORDER BY id DESC LIMIT 5000").fetchall()
    titles = TitleIndex([r["title"] for r in recent])
    budget = PageBudget()
    added = 0
    feeds = conn.execute(
        "SELECT * FROM feeds WHERE muted = 0 AND removed = 0 AND url NOT LIKE ? ORDER BY id",
        (config.DEMO_URL_PREFIX + "%",),
    ).fetchall()
    for feed in feeds:
        entries = fetch_feed(conn, feed)
        try:
            new = store_new_articles(conn, feed, entries, titles, budget)
        except Exception:  # one bad feed must not sink the run
            log.exception("storing entries failed for %s", feed["name"])
            new = 0
        log.info("%s: %d entries, %d new", feed["name"], len(entries), new)
        added += new
    index_and_score(conn, embed.get_embedder())
    removed = apply_retention(conn)
    log.info("run complete: %d new articles, %d expired", added, removed)
    conn.close()
    return added


def check_feeds() -> int:
    """Print whether each feed in feeds.yaml answers with a valid feed."""
    failures = 0
    for seed in config.load_feed_seeds():
        try:
            resp = download_feed(seed.url)
            if resp.status != 200:
                raise ValueError(f"HTTP {resp.status}")
            entries = parse_feed(resp.content)
            with_image = sum(1 for e in entries if entry_image(e))
            print(f"OK    {seed.name:<16} {len(entries):>4} entries, {with_image:>3} with feed image  {seed.url}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {seed.name:<16} {type(exc).__name__}: {exc}  {seed.url}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch feeds and rank new articles.")
    parser.add_argument("--check", action="store_true", help="only verify the feed URLs in feeds.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.check:
        return 1 if check_feeds() else 0
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

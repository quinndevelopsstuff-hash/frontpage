import feedparser

from core import config, db, extract, fetch
from core.dedupe import TitleIndex

RSS = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/" xmlns:content="http://purl.org/rss/1.0/modules/content/">
<channel><title>Test</title>
<item><title>First story about engines</title><link>https://example.org/a?utm_source=rss</link>
  <description>&lt;p&gt;A summary that is long enough to be kept as the card summary text, well over sixty characters.&lt;/p&gt;</description>
  <media:thumbnail url="https://example.org/a.jpg"/><pubDate>Fri, 13 Nov 2026 10:00:00 GMT</pubDate></item>
<item><title>First story about engines!</title><link>https://example.org/a-copy</link>
  <description>Syndicated copy.</description><pubDate>Fri, 13 Nov 2026 08:00:00 GMT</pubDate></item>
<item><title>Second story, quite different</title><link>https://example.org/b</link>
  <description>Short.</description>
  <enclosure url="https://example.org/b.png" type="image/png" length="1"/><pubDate>Fri, 13 Nov 2026 09:00:00 GMT</pubDate></item>
<item><title>No link</title></item>
</channel></rss>"""


def _feed(conn, **overrides):
    seed = config.FeedSeed(url="https://example.org/feed", name="Test", default_topic="CS", **overrides)
    fetch.sync_feeds(conn, [seed])
    return conn.execute("SELECT * FROM feeds WHERE url = ?", (seed.url,)).fetchone()


def _no_network(monkeypatch):
    monkeypatch.setattr(extract, "fetch_page", lambda url: None)
    monkeypatch.setattr(extract, "cache_image", lambda url: None)


def test_entries_are_parsed_canonicalised_and_deduped(conn, monkeypatch):
    _no_network(monkeypatch)
    monkeypatch.setattr(db, "now", lambda: 1794650000.0)  # 2026-11-14
    feed = _feed(conn)
    entries = fetch.parse_feed(RSS)
    budget = fetch.PageBudget(pause=0)
    assert fetch.store_new_articles(conn, feed, entries, TitleIndex(), budget) == 2
    rows = {r["url"]: r for r in conn.execute("SELECT * FROM articles")}
    assert set(rows) == {"https://example.org/a", "https://example.org/b"}  # utm stripped, near-dup title dropped
    assert rows["https://example.org/a"]["image_url"] == "https://example.org/a.jpg"
    assert rows["https://example.org/b"]["image_url"] == "https://example.org/b.png"
    assert rows["https://example.org/a"]["summary"].startswith("A summary that is long enough")
    # a second run (title index rebuilt from the database, as run() does) adds nothing
    known = TitleIndex([r["title"] for r in conn.execute("SELECT title FROM articles")])
    assert fetch.store_new_articles(conn, feed, entries, known, budget) == 0


def test_invalid_feed_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        fetch.parse_feed(b"<html><body>Not a feed</body></html>")
    with pytest.raises(ValueError):
        fetch.parse_feed(b"")
    empty = b'<?xml version="1.0"?><rss version="2.0"><channel><title>Quiet</title></channel></rss>'
    assert fetch.parse_feed(empty) == []  # a real feed with no items is fine


def test_one_failing_feed_is_recorded_not_raised(conn, monkeypatch):
    feed = _feed(conn)

    def boom(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(fetch, "download_feed", boom)
    assert fetch.fetch_feed(conn, feed) == []
    row = conn.execute("SELECT last_error, last_fetch_at, last_ok_at FROM feeds").fetchone()
    assert "timed out" in row["last_error"] and row["last_fetch_at"] and row["last_ok_at"] is None


def test_conditional_get_headers_and_304(conn, monkeypatch):
    feed = _feed(conn)
    seen = {}

    def fake_get(url, *, max_bytes, headers=None, timeout=None):
        seen.update(headers or {})
        if "If-None-Match" in (headers or {}):
            return fetch.net.Response(304, url, {}, b"")
        return fetch.net.Response(200, url, {"etag": '"v1"', "last-modified": "Fri, 13 Nov 2026 10:00:00 GMT"}, RSS)

    monkeypatch.setattr(fetch.net, "safe_get", fake_get)
    assert len(fetch.fetch_feed(conn, feed)) == 4
    feed = conn.execute("SELECT * FROM feeds").fetchone()
    assert feed["etag"] == '"v1"' and feed["last_error"] is None
    assert fetch.fetch_feed(conn, feed) == []  # 304 Not Modified
    assert seen["If-None-Match"] == '"v1"' and seen["If-Modified-Since"].startswith("Fri, 13 Nov")
    assert conn.execute("SELECT etag FROM feeds").fetchone()[0] == '"v1"'


def test_sync_keeps_ui_flags_and_applies_yaml_edits(conn):
    feed = _feed(conn)
    conn.execute("UPDATE feeds SET muted = 1 WHERE id = ?", (feed["id"],))
    conn.commit()
    feed = _feed(conn, weight=0.5)
    assert feed["muted"] == 1 and feed["weight"] == 0.5
    assert conn.execute("SELECT COUNT(*) FROM feeds").fetchone()[0] == 1


def test_retention_keeps_saved_and_feedback(conn):
    feed = _feed(conn)
    now = 1_800_000_000.0
    old = now - 31 * 86400
    for i, published in enumerate([old, old, now - 86400]):
        conn.execute(
            "INSERT INTO articles (feed_id, url, title, published_at, fetched_at, topic_id) VALUES (?, ?, ?, ?, ?, ?)",
            (feed["id"], f"https://example.org/{i}", f"Article {i}", published, published, feed["topic_id"]),
        )
    conn.execute("INSERT INTO feedback (article_id, kind, created_at) VALUES (1, 'save', ?)", (old,))
    conn.commit()
    assert fetch.apply_retention(conn, now=now) == 1
    assert [r[0] for r in conn.execute("SELECT id FROM articles ORDER BY id")] == [1, 3]


def test_entry_image_prefers_thumbnail():
    entry = feedparser.parse(RSS).entries[0]
    assert fetch.entry_image(entry) == "https://example.org/a.jpg"


def test_pick_summary_replaces_boilerplate():
    text = "The real article text goes on for a while and makes a good summary for the card."
    assert fetch.pick_summary("Article URL: https://x Comments URL: https://y", text) == text
    assert fetch.pick_summary("", text) == text
    long_summary = "A proper feed summary that is clearly long enough to be used as it is on the card."
    assert fetch.pick_summary(long_summary, text) == long_summary


def test_og_image_and_text_helpers():
    page = '<html><head><meta content="/img/cover.jpg" property="og:image"></head></html>'
    assert extract.find_og_image(page, "https://example.org/post/1") == "https://example.org/img/cover.jpg"
    assert extract.find_og_image("<html></html>", "https://example.org/") is None
    assert extract.html_to_text("<p>Hello&nbsp;<b>world</b> &amp; all</p>") == "Hello world & all"


def test_full_feed_content_becomes_reader_html(conn, monkeypatch):
    _no_network(monkeypatch)
    paragraph = "A long paragraph of real article text that keeps going for a good while. " * 14
    rss = f"""<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel><title>T</title>
<item><title>Full content story</title><link>https://example.org/full</link>
<description>Short teaser that is long enough to be shown on the card as a summary.</description>
<content:encoded><![CDATA[<p>{paragraph}</p><script>alert(1)</script><p onclick="x()">{paragraph}</p>
<img src="/pics/a.jpg">]]></content:encoded>
<media:thumbnail xmlns:media="http://search.yahoo.com/mrss/" url="https://example.org/t.jpg"/>
<pubDate>Fri, 13 Nov 2026 10:00:00 GMT</pubDate></item></channel></rss>""".encode()
    monkeypatch.setattr(db, "now", lambda: 1794650000.0)
    page_downloads = []
    monkeypatch.setattr(extract, "fetch_page", lambda url, **kw: page_downloads.append(url))
    feed = _feed(conn)
    fetch.store_new_articles(conn, feed, fetch.parse_feed(rss), TitleIndex(), fetch.PageBudget(pause=0))
    row = conn.execute("SELECT content_html, reader_state FROM articles").fetchone()
    assert row["reader_state"] == "remote" and page_downloads == []  # no extra request
    assert "alert" not in row["content_html"] and "onclick" not in row["content_html"]
    assert 'src="https://example.org/pics/a.jpg"' in row["content_html"]  # absolute, not yet cached


def test_short_feed_without_page_leaves_reader_for_later(conn, monkeypatch):
    _no_network(monkeypatch)
    monkeypatch.setattr(db, "now", lambda: 1794650000.0)
    feed = _feed(conn, extract=False)
    fetch.store_new_articles(conn, feed, fetch.parse_feed(RSS), TitleIndex(), fetch.PageBudget(pause=0))
    states = {r[0] for r in conn.execute("SELECT reader_state FROM articles")}
    assert states == {None}  # extracted lazily on first open

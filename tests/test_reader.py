"""Reader view: sanitiser, extraction and its fallback, image localisation, the page."""
import os
import time

import pytest

from core import config, db, extract, reader

BASE = "https://news.example.org/2026/11/story"
LONG = "Plenty of words in this paragraph so that the article counts as real content. " * 6


# --- sanitiser ---------------------------------------------------------------


def clean(html):
    return reader.sanitize_html(html, BASE)


def test_scripts_and_their_content_are_removed():
    out = clean("<p>ok</p><script>alert('x')</script><SCRIPT src=//evil.js></SCRIPT><style>p{}</style>")
    assert out == "<p>ok</p>"


def test_event_handlers_are_stripped():
    out = clean(
        '<p onclick="steal()" onmouseover="x()">hi</p>'
        '<img src="https://cdn.example.org/a.jpg" onerror="alert(1)" onload="x()">'
        '<a href="https://example.org/" onfocus="x()">link</a>'
    )
    for handler in ("onclick", "onmouseover", "onerror", "onload", "onfocus", "steal", "alert"):
        assert handler not in out
    assert '<img src="https://cdn.example.org/a.jpg"' in out


@pytest.mark.parametrize(
    "dangerous",
    [
        '<a href="javascript:alert(1)">x</a>',
        '<a href=" JaVaScRiPt:alert(1)">x</a>',
        '<a href="vbscript:msgbox(1)">x</a>',
        '<a href="data:text/html,<script>alert(1)</script>">x</a>',
    ],
)
def test_script_urls_are_removed(dangerous):
    out = clean(dangerous)
    assert "href" not in out and "script" not in out.lower()


def test_embeds_forms_and_styles_are_removed():
    out = clean(
        '<iframe src="https://evil.example/"></iframe><object data="x"></object><embed src="x">'
        '<form action="/sources/1/remove"><button>Go</button></form><svg><script>alert(1)</script></svg>'
        '<p style="position:fixed" class="ad" id="main">text</p><meta http-equiv="refresh" content="0;url=x">'
        '<base href="https://evil.example/">'
    )
    assert out == "<p>text</p>"


def test_data_and_non_http_images_are_dropped():
    out = clean('<p>a</p><img src="data:image/png;base64,AAAA"><img src="file:///etc/passwd"><img alt="no src">')
    assert out == "<p>a</p>"


def test_structure_is_kept():
    html = (
        "<h2>Head</h2><p>Para <strong>b</strong> <em>i</em> <code>c</code></p>"
        "<ul><li>one</li></ul><ol start=\"3\"><li>three</li></ol><blockquote>q</blockquote>"
        "<figure><img src=\"https://cdn.example.org/a.jpg\" alt=\"A\"><figcaption>cap</figcaption></figure>"
    )
    out = clean(html)
    for fragment in ("<h2>Head</h2>", "<strong>b</strong>", "<em>i</em>", "<code>c</code>", "<ul><li>one</li></ul>",
                     '<ol start="3">', "<blockquote>q</blockquote>", "<figcaption>cap</figcaption>", 'alt="A"'):
        assert fragment in out


def test_links_are_absolute_and_open_safely():
    out = clean('<a href="/other">rel</a> <a href="//cdn.example.org/x">proto</a> <a href="#top">frag</a>')
    assert 'href="https://news.example.org/other"' in out
    assert 'href="https://cdn.example.org/x"' in out
    assert 'href="https://news.example.org/2026/11/story#top"' in out
    assert out.count('target="_blank"') == 3 and out.count('rel="noopener noreferrer nofollow"') == 3


def test_images_load_lazily_without_referrer():
    out = clean('<img src="pics/a.jpg">')
    assert 'src="https://news.example.org/2026/11/pics/a.jpg"' in out
    assert 'loading="lazy"' in out and 'referrerpolicy="no-referrer"' in out


def test_only_local_image_cache_paths_may_be_relative_to_the_app():
    local = "/img/" + "a" * 24 + ".webp"
    assert f'src="{local}"' in clean(f'<img src="{local}">')
    # anything else that looks app-relative is resolved against the article's site
    out = clean('<img src="/img/../sources"><a href="/sources/1/remove">x</a>')
    assert 'src="https://news.example.org/sources"' in out
    assert 'href="https://news.example.org/sources/1/remove"' in out


def _structure(html):
    """Tags, attributes (order-insensitive) and text, for comparing HTML."""
    from html.parser import HTMLParser

    events = []

    class Parser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            events.append((tag, tuple(sorted(attrs))))

        def handle_endtag(self, tag):
            events.append(("/" + tag,))

        def handle_data(self, data):
            events.append(("text", data))

    Parser().feed(html)
    return events


def test_sanitising_is_idempotent():
    # nh3 appends the forced attributes in no fixed order, so compare structure
    once = clean('<p onclick="x">a <a href="/b">b</a></p><img src="c.jpg"><script>1</script>')
    assert _structure(clean(once)) == _structure(once)


# --- extraction and fallback -------------------------------------------------


def page(body):
    return f"<html><head><title>T</title></head><body><nav>Menu Home About</nav><article>{body}</article></body></html>"


def test_extract_html_keeps_article_structure():
    html = reader.extract_html(
        page(f"<h1>The Title Of The Story</h1><p>{LONG}</p><h2>Part two</h2><p>{LONG}</p>"
             "<ul><li>first point</li><li>second point</li></ul><blockquote>A quoted line.</blockquote>"
             "<script>alert(1)</script>"),
        BASE,
        "The title of the story",
    )
    assert html is not None
    assert "<h2>Part two</h2>" in html and "<li>first point</li>" in html and "<blockquote>" in html
    assert "script" not in html and "Menu Home" not in html
    assert "The Title Of The Story" not in html  # repeated title removed; the page shows it already


def test_short_extraction_counts_as_failed():
    assert reader.extract_html(page("<p>Subscribe to read this article.</p>"), BASE, "x") is None
    assert reader.extract_html("", BASE, "x") is None
    assert reader.extract_html("<html><body></body></html>", BASE, "x") is None
    assert reader.prepare_html("<p>" + "x " * 100 + "</p>", BASE, "t") is None  # 200 chars < 300


def test_extract_html_survives_garbage():
    assert reader.extract_html("\x00\x01<<<>>>" * 1000, BASE, "x") is None


# --- lazy backfill -----------------------------------------------------------


def _article(conn, **values):
    conn.execute(
        "INSERT OR IGNORE INTO feeds (id, url, name, default_topic, created_at) "
        "VALUES (1, 'https://news.example.org/feed', 'Example News', 'CS', 0)"
    )
    fields = {"feed_id": 1, "url": BASE, "title": "The title of the story", "published_at": time.time(),
              "fetched_at": time.time(), "topic": "CS", "summary": "The RSS summary of the story."}
    fields.update(values)
    cur = conn.execute(
        f"INSERT INTO articles ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})", list(fields.values())
    )
    conn.commit()
    return conn.execute("SELECT * FROM articles WHERE id = ?", (cur.lastrowid,)).fetchone()


def _row(conn, article_id):
    return conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()


def test_backfill_extracts_on_first_open_and_caches_result(conn, monkeypatch):
    calls = []
    good = page(f"<p>{LONG}</p><p>{LONG}</p>")
    monkeypatch.setattr(extract, "fetch_page", lambda url, total_timeout=None: calls.append(url) or (good, url))
    article = _article(conn)
    result = reader.ensure_reader_content(conn, article)
    assert result.state == "ready" and "Plenty of words" in result.html
    assert _row(conn, article["id"])["reader_state"] == "ready"
    again = reader.ensure_reader_content(conn, _row(conn, article["id"]))
    assert again.html == result.html and calls == [BASE]  # downloaded once


def test_failed_download_falls_back_and_retries_later(conn, monkeypatch):
    monkeypatch.setattr(extract, "fetch_page", lambda url, total_timeout=None: None)
    article = _article(conn)
    result = reader.ensure_reader_content(conn, article)
    assert result.html is None and result.state == "failed"
    row = _row(conn, article["id"])
    assert row["reader_state"] == "failed" and row["content_html"] is None

    good = page(f"<p>{LONG}</p><p>{LONG}</p>")
    monkeypatch.setattr(extract, "fetch_page", lambda url, total_timeout=None: (good, url))
    assert reader.ensure_reader_content(conn, row).html is None  # too soon to retry
    conn.execute("UPDATE articles SET reader_checked_at = ? WHERE id = ?",
                 (time.time() - reader.READER_RETRY_SECONDS - 1, article["id"]))
    assert reader.ensure_reader_content(conn, _row(conn, article["id"])).state == "ready"


def test_short_text_falls_back(conn, monkeypatch):
    paywall = page("<p>Please subscribe to continue reading.</p>")
    monkeypatch.setattr(extract, "fetch_page", lambda url, total_timeout=None: (paywall, url))
    assert reader.ensure_reader_content(conn, _article(conn)).html is None


def test_images_go_through_the_cache(conn, monkeypatch):
    cached = {"https://cdn.example.org/one.jpg": "1" * 24 + ".webp"}
    attempted = []

    def fake_cache(url, min_size=(200, 100), total_timeout=None):
        attempted.append(url)
        return cached.get(url)

    monkeypatch.setattr(extract, "cache_image", fake_cache)
    html = (
        f"<p>{LONG}</p>"
        '<img src="https://cdn.example.org/hero.jpg" alt="hero">'
        '<img src="https://cdn.example.org/one.jpg" alt="cached">'
        '<img src="https://tracker.example.net/pixel.gif" alt="">'
    )
    article = _article(conn, content_html=html, reader_state="remote",
                       image_url="https://cdn.example.org/hero.jpg", image_path="h" * 24 + ".webp")
    result = reader.ensure_reader_content(conn, article)
    assert result.state == "ready"
    assert f'src="/img/{"1" * 24}.webp"' in result.html  # rewritten to the local cache
    assert "pixel.gif" not in result.html  # tried and failed -> dropped
    assert "hero.jpg" not in result.html  # duplicate of the hero shown above the text
    assert "https://cdn.example.org/hero.jpg" not in attempted


def test_images_beyond_the_cap_stay_remote(monkeypatch):
    monkeypatch.setattr(extract, "cache_image", lambda url, **kw: None)
    urls = [f"https://cdn.example.org/{i}.jpg" for i in range(reader.MAX_INLINE_IMAGES + 2)]
    html = f"<p>{LONG}</p>" + "".join(f'<img src="{u}">' for u in urls)
    out = reader.localize_images(reader.sanitize_html(html, BASE), BASE)
    assert [u for u in urls if u in out] == urls[reader.MAX_INLINE_IMAGES:]


def test_sweep_deletes_only_unreferenced_old_images(conn):
    images = config.images_dir()
    names = {k: c * 24 + ".webp" for k, c in zip(("hero", "inline", "orphan", "young"), "abcd")}
    for name in names.values():
        (images / name).write_bytes(b"x")
    old = time.time() - 2 * 86400
    for key in ("hero", "inline", "orphan"):
        os.utime(images / names[key], (old, old))
    _article(conn, image_path=names["hero"], reader_state="ready",
             content_html=f'<p>x</p><img src="/img/{names["inline"]}">')
    assert reader.sweep_images(conn) == 1
    assert sorted(p.name for p in images.iterdir()) == sorted([names["hero"], names["inline"], names["young"]])


def test_migration_adds_reader_columns_to_an_existing_database(tmp_path):
    path = tmp_path / "v1.db"
    import sqlite3

    raw = sqlite3.connect(path)
    db.migrate(raw, db.MIGRATIONS[:1])
    raw.execute("INSERT INTO feeds (url, name, default_topic, created_at) VALUES ('https://e.org/f', 'F', 'CS', 0)")
    raw.execute("INSERT INTO articles (feed_id, url, title, published_at, fetched_at, topic) "
                "VALUES (1, 'https://e.org/a', 'A', 0, 0, 'CS')")
    raw.commit()
    raw.close()
    conn = db.connect(path)
    row = conn.execute("SELECT title, content_html, reader_state FROM articles").fetchone()
    assert tuple(row) == ("A", None, None)
    assert db.schema_version(conn) == len(db.MIGRATIONS)


# --- the page ----------------------------------------------------------------


def _id(client, fragment):
    with client.application.app_context():
        c = client.application.extensions["get_db"]()
        return c.execute("SELECT id FROM articles WHERE title LIKE ?", (f"%{fragment}%",)).fetchone()[0]


def _kinds(client, article_id):
    with client.application.app_context():
        c = client.application.extensions["get_db"]()
        return {r[0] for r in c.execute("SELECT kind FROM feedback WHERE article_id = ?", (article_id,))}


def test_cards_link_to_the_reader(client):
    html = client.get("/").get_data(as_text=True)
    assert 'href="/read/' in html and 'target="_blank"' not in html


def test_reader_page_renders_and_records_open(client):
    article_id = _id(client, "engine race")
    resp = client.get(f"/read/{article_id}", headers={"Referer": "http://localhost/?tab=F1"})
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200 and resp.headers["Cache-Control"] == "no-store"
    assert '<div class="prose">' in html and "What the reader keeps" in html
    assert "How a rule change could reshape" in html and 'class="topic">F1<' in html
    assert 'data-back' in html and 'href="/?tab=F1"' in html  # Back without JS returns to the tab
    assert 'aria-label="More like this"' in html and "View original" in html
    assert "<script>" not in html.split("</main>")[0]
    assert _kinds(client, article_id) == {"open"}
    with client.application.app_context():  # the open retrained the model
        c = client.application.extensions["get_db"]()
        assert c.execute("SELECT value FROM kv WHERE key = 'model'").fetchone() is not None


def test_reader_shows_feedback_state(client):
    article_id = _id(client, "tyre wear")
    client.post("/api/feedback", json={"article_id": article_id, "kind": "save"})
    html = client.get(f"/read/{article_id}").get_data(as_text=True)
    assert 'data-kind="save" aria-pressed="true"' in html
    assert 'data-kind="up" aria-pressed="false"' in html


def test_reader_fallback_page(client, monkeypatch):
    monkeypatch.setattr(extract, "fetch_page", lambda url, total_timeout=None: None)
    article_id = _id(client, "flexible wings")  # the demo article without a reader body
    html = client.get(f"/read/{article_id}").get_data(as_text=True)
    assert "Open original" in html and "New load tests and rule changes" in html
    assert '<div class="prose">' not in html
    assert f'href="https://example.com/frontpage-demo/' in html


def test_reader_backfills_untried_articles(client, monkeypatch):
    article_id = _id(client, "scheduling idea")
    with client.application.app_context():
        c = client.application.extensions["get_db"]()
        c.execute("UPDATE articles SET content_html = NULL, reader_state = NULL WHERE id = ?", (article_id,))
        c.commit()
    good = page(f"<p>{LONG}</p><p>{LONG}</p><script>alert(1)</script><img src=x onerror=alert(1)>")
    monkeypatch.setattr(extract, "fetch_page", lambda url, total_timeout=None: (good, url))
    monkeypatch.setattr(extract, "cache_image", lambda url, **kw: None)
    html = client.get(f"/read/{article_id}").get_data(as_text=True)
    assert "Plenty of words" in html and "alert" not in html and "onerror" not in html


def test_stored_html_is_sanitised_again_on_render(client):
    article_id = _id(client, "battery pack")
    with client.application.app_context():
        c = client.application.extensions["get_db"]()
        c.execute("UPDATE articles SET content_html = ? WHERE id = ?",
                  (f"<p>{LONG}</p><script>alert(1)</script><p onclick='x()'>tail</p>", article_id))
        c.commit()
    html = client.get(f"/read/{article_id}").get_data(as_text=True)
    assert "alert(1)" not in html and "onclick" not in html and "<p>tail</p>" in html


def test_reader_404_and_security_headers(client):
    assert client.get("/read/999999").status_code == 404
    csp = client.get("/").headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "frame-src 'none'" in csp and "object-src 'none'" in csp

"""Dynamic topics: seeding, validation, the migration, assignment, the UI endpoints."""
import sqlite3
import time

import numpy as np
import pytest

from core import colors, config, db, embed, rank, topics


def seed(name, description="About it.", accent="#60A5FA", interests=None):
    return config.TopicSeed(name, description, accent, interests or {})


def names(conn):
    return [t.name for t in topics.list_topics(conn)]


# --- seeding ------------------------------------------------------------------


def test_fresh_database_is_seeded_from_interests_yaml(conn):
    assert names(conn) == ["F1", "AI", "CS", "Engineering", "Philosophy"]
    philosophy = topics.by_slug(conn, "philosophy")
    assert philosophy.accent_color == "#2DD4BF"
    for idea in ("mind", "ethics", "Stoicism", "existentialism", "artificial intelligence"):
        assert idea in philosophy.description
    assert "Stoicism" in philosophy.interests
    assert topics.by_slug(conn, "f1").accent_color == "#FF6B6B"


def test_seeding_updates_untouched_topics_but_never_edited_ones(conn):
    f1, cs = topics.by_slug(conn, "f1"), topics.by_slug(conn, "cs")
    topics.update_topic(conn, cs.id, "Computing", "My own words about computing.", "#4ADE80")
    yaml_now = config.load_topics()
    changed = [
        seed(s.name, s.description + " (edited in yaml)", "#F472B6", s.interests) if s.name in ("F1", "CS") else s
        for s in yaml_now
    ]
    topics.seed_topics(conn, changed)
    f1_after, cs_after = topics.get(conn, f1.id), topics.get(conn, cs.id)
    assert f1_after.description.endswith("(edited in yaml)") and f1_after.accent_color == "#F472B6"
    assert (cs_after.name, cs_after.description, cs_after.accent_color) == (
        "Computing", "My own words about computing.", "#4ADE80"
    )
    assert "CS" not in names(conn)  # the renamed topic was not re-created


def test_seeding_adds_new_yaml_topics_and_respects_ui_ones(conn):
    mine = topics.create_topic(conn, "Photography", "Cameras and photos.", "#FB923C")
    topics.seed_topics(conn, config.load_topics() + [seed("Space", "Rockets."), seed("photography", "yaml words")])
    assert names(conn)[-2:] == ["Photography", "Space"]
    assert topics.get(conn, mine.id).description == "Cameras and photos."  # adopted, not overwritten
    assert names(conn).count("Photography") == 1


def test_deleted_seed_topics_stay_deleted(conn):
    philosophy = topics.by_slug(conn, "philosophy")
    topics.delete_topic(conn, philosophy.id, move_feeds_to=None)
    topics.seed_topics(conn)
    assert "Philosophy" not in names(conn)


def test_seeding_is_idempotent(conn):
    before = [(t.id, t.name, t.position) for t in topics.list_topics(conn)]
    assert topics.seed_topics(conn) == 0
    assert [(t.id, t.name, t.position) for t in topics.list_topics(conn)] == before


# --- validation -----------------------------------------------------------------


@pytest.mark.parametrize("duplicate", ["Photography", "photography", "  PHOTOGRAPHY ", "pHoToGrApHy"])
def test_duplicate_names_are_rejected_case_insensitively(conn, duplicate):
    topics.create_topic(conn, "Photography", "Cameras.", "#FB923C")
    with pytest.raises(topics.TopicError, match="already a topic"):
        topics.create_topic(conn, duplicate, "Again.", "#F472B6")


def test_duplicate_of_a_seeded_topic_is_rejected(conn):
    with pytest.raises(topics.TopicError):
        topics.create_topic(conn, "philosophy", "Again.", "#F472B6")


@pytest.mark.parametrize("bad", ["", "   ", "!!!", "x" * 31, "Saved", "For You", "for-you"])
def test_empty_reserved_and_long_names_are_rejected(conn, bad):
    with pytest.raises(topics.TopicError):
        topics.create_topic(conn, bad, "Description.", "#FB923C")


def test_description_and_colour_are_required(conn):
    with pytest.raises(topics.TopicError, match="Describe"):
        topics.create_topic(conn, "Photography", "   ", "#FB923C")
    with pytest.raises(topics.TopicError, match="swatch"):
        topics.create_topic(conn, "Photography", "Cameras.", "#123456")


def test_rename_rejects_another_topics_name_but_allows_recasing_its_own(conn):
    cs = topics.by_slug(conn, "cs")
    with pytest.raises(topics.TopicError):
        topics.update_topic(conn, cs.id, "f1", cs.description, cs.accent_color)
    renamed = topics.update_topic(conn, cs.id, "cs", cs.description, cs.accent_color)
    assert renamed.name == "cs" and renamed.slug == "cs"


def test_presets_pass_wcag_aa_on_the_card():
    assert len(colors.PRESET_ACCENTS) == 8
    for label, hex_ in colors.PRESET_ACCENTS:
        assert colors.contrast_ratio(hex_, colors.CARD_BG) >= 4.5, label
        cover_start = colors.cover_gradient(hex_).split(",")[0]
        assert colors.contrast_ratio(hex_, cover_start) >= 3.0, label  # large cover headline


# --- migration ---------------------------------------------------------------------


def test_migration_moves_topic_names_to_the_topics_table(tmp_path):
    path = tmp_path / "v2.db"
    raw = sqlite3.connect(path)
    raw.row_factory = sqlite3.Row
    db.migrate(raw, db.MIGRATIONS[:2])  # a database from before dynamic topics
    raw.executemany(
        "INSERT INTO feeds (url, name, default_topic, created_at) VALUES (?, ?, ?, 0)",
        [("https://a/f", "A", "F1"), ("https://b/f", "B", "Gardening"), ("https://c/f", "C", "ai")],
    )
    raw.executemany(
        "INSERT INTO articles (feed_id, url, title, published_at, fetched_at, topic) VALUES (?, ?, ?, 0, 0, ?)",
        [(1, "https://a/1", "One", "F1"), (1, "https://a/2", "Two", "Engineering"), (2, "https://b/1", "Three", "Gardening")],
    )
    raw.commit()
    raw.close()

    conn = db.connect(path)
    assert db.schema_version(conn) == len(db.MIGRATIONS)
    feed_topics = dict(conn.execute("SELECT f.name, t.name FROM feeds f JOIN topics t ON t.id = f.topic_id"))
    assert feed_topics == {"A": "F1", "B": "Gardening", "C": "AI"}  # unknown name kept as its own topic
    article_topics = dict(conn.execute("SELECT a.title, t.name FROM articles a JOIN topics t ON t.id = a.topic_id"))
    assert article_topics == {"One": "F1", "Two": "Engineering", "Three": "Gardening"}
    feed_cols = {r[1] for r in conn.execute("PRAGMA table_info(feeds)")}
    article_cols = {r[1] for r in conn.execute("PRAGMA table_info(articles)")}
    assert "default_topic" not in feed_cols and "topic" not in article_cols
    assert "Philosophy" in names(conn)


def test_migration_rolls_back_if_it_fails(tmp_path, monkeypatch):
    path = tmp_path / "v2b.db"
    raw = sqlite3.connect(path)
    db.migrate(raw, db.MIGRATIONS[:2])
    raw.close()
    monkeypatch.setattr(topics, "seed_topics", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        db.connect(path)
    check = sqlite3.connect(path)
    assert check.execute("SELECT version FROM schema_version").fetchone()[0] == 2
    assert not check.execute("SELECT name FROM sqlite_master WHERE name = 'topics'").fetchone()
    assert "default_topic" in {r[1] for r in check.execute("PRAGMA table_info(feeds)")}


# --- topic assignment for a new topic ------------------------------------------------

PHOTO_DESCRIPTION = "Photography cameras lenses aperture exposure shutter sensor photographers photo editing"
PHOTO_ARTICLE = "Choosing camera lenses: aperture, exposure, shutter speed and sensor size for photographers"


def _article(conn, title, topic_name="CS"):
    topic_id = topics.id_by_name(conn, topic_name)
    conn.execute(
        "INSERT OR IGNORE INTO feeds (id, url, name, topic_id, created_at) VALUES (900, 'https://x/f', 'X', ?, 0)",
        (topic_id,),
    )
    cur = conn.execute(
        "INSERT INTO articles (feed_id, url, title, published_at, fetched_at, topic_id, content_text) "
        "VALUES (900, ?, ?, ?, ?, ?, ?)",
        (f"https://x/{abs(hash(title))}", title, time.time(), time.time(), topic_id, title),
    )
    conn.commit()
    return cur.lastrowid


def _topic_of(conn, article_id):
    return conn.execute(
        "SELECT t.name FROM articles a JOIN topics t ON t.id = a.topic_id WHERE a.id = ?", (article_id,)
    ).fetchone()[0]


def _index(conn):
    from core import fetch

    fetch.index_and_score(conn, embed.get_embedder())


def test_new_topic_is_used_for_new_articles_without_restart(conn):
    old = _article(conn, PHOTO_ARTICLE + " (old)")
    _index(conn)
    assert _topic_of(conn, old) == "CS"
    topics.create_topic(conn, "Photography", PHOTO_DESCRIPTION, "#FB923C")
    moved = rank.refresh_anchors(conn, embed.get_embedder(), reclassify_existing=False)
    assert moved == 0 and _topic_of(conn, old) == "CS"  # existing articles stay without the checkbox
    new = _article(conn, PHOTO_ARTICLE)
    _index(conn)
    assert _topic_of(conn, new) == "Photography"


def test_reclassify_moves_only_articles_still_in_their_feed_topic(conn):
    photo = _article(conn, PHOTO_ARTICLE)
    other = _article(conn, "A database engine for time series storage and query performance")
    _index(conn)
    engineering = topics.id_by_name(conn, "Engineering")
    moved_by_hand = _article(conn, PHOTO_ARTICLE + " again")
    _index(conn)
    conn.execute("UPDATE articles SET topic_id = ? WHERE id = ?", (engineering, moved_by_hand))
    conn.commit()
    topics.create_topic(conn, "Photography", PHOTO_DESCRIPTION, "#FB923C")
    moved = rank.refresh_anchors(conn, embed.get_embedder(), reclassify_existing=True)
    assert moved == 1
    assert (_topic_of(conn, photo), _topic_of(conn, other), _topic_of(conn, moved_by_hand)) == (
        "Photography", "CS", "Engineering"
    )


def test_only_the_new_topics_sentences_are_embedded(conn):
    class Counting(embed.HashEmbedder):
        calls: list = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return super().embed(texts)

    counting = Counting()
    rank.load_interests(conn, counting)
    counting.calls.clear()
    topics.create_topic(conn, "Photography", PHOTO_DESCRIPTION, "#FB923C")
    _, changed = rank.load_interests(conn, counting)
    assert changed and counting.calls == [[f"Photography: {PHOTO_DESCRIPTION}"]]


def test_new_topic_anchor_feeds_the_prior(conn):
    article = _article(conn, PHOTO_ARTICLE)
    _index(conn)
    before = conn.execute("SELECT prior FROM articles WHERE id = ?", (article,)).fetchone()[0]
    topics.create_topic(conn, "Photography", PHOTO_DESCRIPTION, "#FB923C")
    rank.refresh_anchors(conn, embed.get_embedder())
    row = conn.execute("SELECT prior, prior_label FROM articles WHERE id = ?", (article,)).fetchone()
    assert row["prior"] > before and row["prior_label"] == "Photography"


# --- web endpoints -------------------------------------------------------------------


def _db(client):
    return client.application.extensions["get_db"]()


def test_tabs_and_dropdown_come_from_the_table(client):
    html = client.get("/").get_data(as_text=True)
    for name, slug in (("F1", "f1"), ("Philosophy", "philosophy")):
        assert f'href="/?tab={slug}"' in html and f">{name}</a>" in html
    sources = client.get("/sources").get_data(as_text=True)
    assert ">Philosophy</option>" in sources and '<option value="__new__">+ Create new topic…</option>' in sources
    assert client.get("/?tab=PHILOSOPHY").get_data(as_text=True).count('aria-current="page">Philosophy') == 1


def test_create_topic_api(client):
    resp = client.post(
        "/api/topics",
        json={"name": "Photography", "description": PHOTO_DESCRIPTION, "accent": "#FB923C", "reclassify": False},
    )
    assert resp.status_code == 201
    data = resp.json
    assert data["slug"] == "photography" and data["href"] == "/?tab=photography" and data["embedded"] is True
    assert 'href="/?tab=photography"' in client.get("/").get_data(as_text=True)  # in the tabs immediately
    dup = client.post("/api/topics", json={"name": "PHOTOGRAPHY", "description": "x", "accent": "#FB923C"})
    assert dup.status_code == 400 and "already a topic" in dup.json["error"]
    empty = client.post("/api/topics", json={"name": "  ", "description": "x", "accent": "#FB923C"})
    assert empty.status_code == 400
    assert client.post("/api/topics", data={"name": "Form post"}).status_code == 400


def test_add_feed_with_a_new_topic_without_javascript(client):
    resp = client.post(
        "/sources/add",
        data={"url": "https://example.net/photo.xml", "name": "Photo Blog", "topic": "__new__",
              "new_topic_name": "Photography", "new_topic_description": PHOTO_DESCRIPTION,
              "new_topic_accent": "#FB923C", "reclassify": "1"},
    )
    assert resp.status_code == 302 and "Created+topic" in resp.headers["Location"]
    with client.application.app_context():
        row = _db(client).execute(
            "SELECT t.name FROM feeds f JOIN topics t ON t.id = f.topic_id WHERE f.url = 'https://example.net/photo.xml'"
        ).fetchone()
        assert row[0] == "Photography"
    bad = client.post("/sources/add", data={"url": "https://example.net/b.xml", "topic": "__new__",
                                            "new_topic_name": "f1", "new_topic_description": "x",
                                            "new_topic_accent": "#FB923C"})
    assert "already+a+topic" in bad.headers["Location"]


def test_edit_and_recolour_a_topic(client):
    with client.application.app_context():
        cs = topics.by_slug(_db(client), "cs")
    resp = client.post(f"/sources/topics/{cs.id}", data={"name": "Computing", "description": "Software and systems.",
                                                         "accent": "#4ADE80"})
    assert resp.status_code == 302 and "error" not in resp.headers["Location"]
    with client.application.app_context():
        after = topics.get(_db(client), cs.id)
    assert (after.name, after.slug, after.accent_color) == ("Computing", "computing", "#4ADE80")
    html = client.get("/?tab=computing").get_data(as_text=True)
    assert 'class="topic">Computing<' in html and "--accent: #4ADE80" in html
    off_palette = client.post(f"/sources/topics/{cs.id}", data={"name": "Computing", "description": "x",
                                                                 "accent": "#000000"})
    assert "error=" in off_palette.headers["Location"]


def test_delete_topic_asks_where_feeds_go(client):
    with client.application.app_context():
        conn = _db(client)
        f1, eng = topics.by_slug(conn, "f1"), topics.by_slug(conn, "engineering")
        f1_feeds = topics.feed_count(conn, f1.id)
    sources = client.get("/sources").get_data(as_text=True)
    assert f"Move its {f1_feeds} feeds to" in sources
    refused = client.post(f"/sources/topics/{f1.id}/delete", data={})
    assert "error=" in refused.headers["Location"]
    client.post(f"/sources/topics/{f1.id}/delete", data={"move_to": str(eng.id)})
    with client.application.app_context():
        conn = _db(client)
        assert topics.get(conn, f1.id) is None
        assert conn.execute("SELECT COUNT(*) FROM feeds WHERE topic_id = ?", (f1.id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM articles WHERE topic_id = ?", (f1.id,)).fetchone()[0] == 0
        assert topics.feed_count(conn, eng.id) >= f1_feeds
    assert 'href="/?tab=f1"' not in client.get("/").get_data(as_text=True)


def test_last_topic_cannot_be_deleted(conn):
    remaining = topics.list_topics(conn)
    for topic in remaining[1:]:
        topics.delete_topic(conn, topic.id, move_feeds_to=remaining[0].id)
    with pytest.raises(topics.TopicError, match="at least one"):
        topics.delete_topic(conn, remaining[0].id, move_feeds_to=None)


def test_no_hardcoded_topic_lists_remain():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    # The demo seeder's fake articles name their topics; everything else must read the table.
    sources = [p for d in ("core", "app") for p in (root / d).rglob("*.py") if p.name != "seed_demo.py"]
    code = "\n".join(p.read_text() for p in sources)
    code += "\n".join(p.read_text() for p in (root / "app").rglob("*.html"))
    code += (root / "app" / "static" / "app.js").read_text()
    assert "TOPIC_STYLES" not in code
    for name in ('"F1"', '"Engineering"', "'F1'", "#7a1620"):
        assert name not in code, name

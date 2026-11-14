def _article(client, fragment):
    with client.application.app_context():
        conn = client.application.extensions["get_db"]()
        return conn.execute("SELECT id FROM articles WHERE title LIKE ?", (f"%{fragment}%",)).fetchone()[0]


def _topic_id(client, name):
    with client.application.app_context():
        conn = client.application.extensions["get_db"]()
        return conn.execute("SELECT id FROM topics WHERE name = ?", (name,)).fetchone()[0]


def _kinds(client, article_id):
    with client.application.app_context():
        conn = client.application.extensions["get_db"]()
        return {r[0] for r in conn.execute("SELECT kind FROM feedback WHERE article_id = ?", (article_id,))}


def test_index_renders_hero_and_cards(client):
    html = client.get("/").get_data(as_text=True)
    assert html.count("<article") == 20
    assert html.count("card--hero") == 1 and "Top Pick" in html
    assert "% match" in html and 'aria-label="More like this"' in html
    assert 'aria-current="page">For You' in html


def test_topic_tab_only_shows_that_topic(client):
    html = client.get("/?tab=F1").get_data(as_text=True)
    assert html.count("<article") == 6
    assert 'class="topic">AI<' not in html


def test_feedback_up_then_down_is_exclusive(client):
    article = _article(client, "engine race")
    resp = client.post("/api/feedback", json={"article_id": article, "kind": "up", "value": True})
    assert resp.status_code == 200
    assert resp.json["up"] is True and resp.json["down"] is False and resp.json["n_labels"] == 1
    resp = client.post("/api/feedback", json={"article_id": article, "kind": "down", "value": True})
    assert resp.json["up"] is False and resp.json["down"] is True
    assert _kinds(client, article) == {"down"}


def test_feedback_toggle_off_and_save(client):
    article = _article(client, "tyre wear")
    client.post("/api/feedback", json={"article_id": article, "kind": "save", "value": True})
    client.post("/api/feedback", json={"article_id": article, "kind": "up", "value": True})
    assert _kinds(client, article) == {"save", "up"}
    resp = client.post("/api/feedback", json={"article_id": article, "kind": "up", "value": False})
    assert resp.json["up"] is False and resp.json["saved"] is True
    assert "tyre wear" in client.get("/?tab=saved").get_data(as_text=True)
    resp = client.post("/api/feedback", json={"article_id": article, "kind": "save", "value": False})
    assert resp.json["saved"] is False and resp.json["n_labels"] == 0
    assert "Nothing saved yet" in client.get("/?tab=saved").get_data(as_text=True)


def test_feedback_retrains_immediately(client):
    article = _article(client, "engine race")
    with client.application.app_context():
        conn = client.application.extensions["get_db"]()
        assert conn.execute("SELECT COUNT(*) FROM articles WHERE model_score IS NOT NULL").fetchone()[0] == 0
    client.post("/api/feedback", json={"article_id": article, "kind": "up"})
    with client.application.app_context():
        conn = client.application.extensions["get_db"]()
        assert conn.execute("SELECT COUNT(*) FROM articles WHERE model_score IS NULL").fetchone()[0] == 0
        assert conn.execute("SELECT value FROM kv WHERE key = 'model'").fetchone() is not None


def test_open_is_recorded_once_and_cannot_be_undone(client):
    article = _article(client, "scheduling idea")
    for _ in range(2):
        assert client.post("/api/feedback", json={"article_id": article, "kind": "open"}).status_code == 200
    client.post("/api/feedback", json={"article_id": article, "kind": "open", "value": False})
    assert _kinds(client, article) == {"open"}


def test_feedback_rejects_bad_input(client):
    article = _article(client, "engine race")
    assert client.post("/api/feedback", json={"article_id": article, "kind": "love"}).status_code == 400
    assert client.post("/api/feedback", json={"article_id": "1", "kind": "up"}).status_code == 400
    assert client.post("/api/feedback", json={"article_id": article, "kind": "up", "value": "yes"}).status_code == 400
    assert client.post("/api/feedback", json={"article_id": 999999, "kind": "up"}).status_code == 404
    # form posts (what a cross-site page could send) are refused
    assert client.post("/api/feedback", data={"article_id": article, "kind": "up"}).status_code == 400
    assert client.post("/api/feedback", json=[1, 2]).status_code == 400
    assert _kinds(client, article) == set()


def test_disliked_article_leaves_the_feed(client):
    article = _article(client, "thermal camera")
    client.post("/api/feedback", json={"article_id": article, "kind": "down"})
    assert "thermal camera" not in client.get("/").get_data(as_text=True)


def test_load_more_api(client):
    import re

    first = re.findall(r'data-id="(\d+)"', client.get("/").get_data(as_text=True))
    assert len(first) == 20
    # feedback between pages reorders the list; nothing may be repeated or lost
    client.post("/api/feedback", json={"article_id": int(first[3]), "kind": "up"})
    data = client.get("/api/articles?tab=for-you&seen=" + ",".join(first)).json
    rest = re.findall(r'data-id="(\d+)"', data["html"])
    assert len(rest) == 6 and not set(rest) & set(first) and data["has_more"] is False
    assert "card--hero" not in data["html"]
    more = client.get("/api/articles?tab=for-you").json
    assert more["html"].count("<article") == 18 and more["has_more"] is True
    assert client.get("/api/articles?tab=nonsense&seen=x,,-5,9").status_code == 200


def test_sources_page_add_mute_remove(client):
    html = client.get("/sources").get_data(as_text=True)
    assert "Demo: Hackaday" in html and "Add a feed" in html
    resp = client.post(
        "/sources/add",
        data={"url": "https://example.net/rss", "name": "New Feed", "topic": str(_topic_id(client, "AI")), "weight": "1.2"},
    )
    assert resp.status_code == 302
    with client.application.app_context():
        conn = client.application.extensions["get_db"]()
        row = conn.execute(
            "SELECT f.*, t.name AS topic FROM feeds f JOIN topics t ON t.id = f.topic_id WHERE url = 'https://example.net/rss'"
        ).fetchone()
        assert (row["name"], row["topic"], row["weight"], row["muted"]) == ("New Feed", "AI", 1.2, 0)
        feed_id = row["id"]
    client.post(f"/sources/{feed_id}/mute")
    assert "Unmute New Feed" in client.get("/sources").get_data(as_text=True)
    client.post(f"/sources/{feed_id}/remove")
    assert "New Feed" not in client.get("/sources").get_data(as_text=True)
    bad = client.post("/sources/add", data={"url": "javascript:alert(1)", "topic": str(_topic_id(client, "AI"))})
    assert "error=" in bad.headers["Location"]


def test_feed_errors_are_shown_on_sources(client):
    with client.application.app_context():
        conn = client.application.extensions["get_db"]()
        conn.execute("UPDATE feeds SET last_error = 'ValueError: HTTP 404' WHERE name = 'Demo: Hackaday'")
        conn.commit()
    assert "Last error: ValueError: HTTP 404" in client.get("/sources").get_data(as_text=True)


def test_pwa_endpoints(client):
    manifest = client.get("/manifest.webmanifest").json
    assert manifest["display"] == "standalone" and len(manifest["icons"]) == 3
    sw = client.get("/sw.js")
    assert sw.headers["Content-Type"].startswith("application/javascript") and "frontpage-shell-" in sw.text
    for icon in ("icon-192.png", "icon-512.png", "apple-touch-icon.png"):
        assert client.get(f"/static/icons/{icon}").status_code == 200
    html = client.get("/").get_data(as_text=True)
    assert 'rel="apple-touch-icon"' in html and 'name="theme-color" content="#0d0f14"' in html
    assert client.get("/healthz").json == {"ok": True}


def test_no_emoji_in_ui(client):
    html = client.get("/").get_data(as_text=True) + client.get("/sources").get_data(as_text=True)
    assert not any(ord(ch) >= 0x1F000 for ch in html)

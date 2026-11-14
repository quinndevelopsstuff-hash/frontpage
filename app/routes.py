"""Routes: the feed pages, the reader, the feedback API and the Sources page."""
from __future__ import annotations

import sqlite3
import time
from urllib.parse import urlsplit

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from core import config, db, rank, reader

bp = Blueprint("main", __name__)

FEEDBACK_KINDS = ("up", "down", "save", "open")


def conn() -> sqlite3.Connection:
    return current_app.extensions["get_db"]()


def fulldate(timestamp: float | None) -> str:
    """'Sat 14 Nov 2026, 10:32' in the Pi's local time."""
    if not timestamp:
        return ""
    return time.strftime("%a %-d %b %Y, %H:%M", time.localtime(timestamp))


def isodate(timestamp: float | None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(timestamp)) if timestamp else ""


def timeago(timestamp: float | None) -> str:
    """'5m ago', '2h ago', '3d ago'."""
    if not timestamp:
        return "never"
    seconds = max(0, time.time() - timestamp)
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def tabs() -> list[dict[str, str]]:
    topics = [{"key": t.name, "label": t.name} for t in config.load_topics()]
    return [{"key": "for-you", "label": "For You"}, *topics, {"key": "saved", "label": "Saved"}]


def valid_tab(value: str | None) -> str:
    keys = {t["key"] for t in tabs()}
    return value if value in keys else "for-you"


def load_cards(tab: str, limit: int, seen: set[int] | None = None) -> tuple[list[dict], bool]:
    """The next `limit` articles of a tab (skipping `seen` ids), ready for the card template."""
    page, has_more = rank.build_feed(conn(), tab, 0, limit, exclude=seen)
    if not page:
        return [], False
    ids = [article_id for article_id, _ in page]
    marks = ",".join("?" * len(ids))
    rows = conn().execute(
        f"SELECT a.id, a.url, a.title, a.summary, a.published_at, a.image_path, a.topic, a.score, a.reason, "
        f"f.name AS source, "
        f"(SELECT GROUP_CONCAT(kind) FROM feedback fb WHERE fb.article_id = a.id) AS kinds "
        f"FROM articles a JOIN feeds f ON f.id = a.feed_id WHERE a.id IN ({marks})",
        ids,
    ).fetchall()
    by_id = {row["id"]: row for row in rows}
    cards = []
    for article_id, is_explore in page:
        row = by_id.get(article_id)
        if row is None:
            continue
        kinds = set((row["kinds"] or "").split(","))
        cards.append(
            {
                **dict(row),
                "match": round((row["score"] or 0) * 100),
                "reason": "from your explore slot" if is_explore else (row["reason"] or ""),
                "explore": is_explore,
                "up": "up" in kinds,
                "down": "down" in kinds,
                "saved": "save" in kinds,
            }
        )
    return cards, has_more


@bp.get("/")
def index():
    tab = valid_tab(request.args.get("tab"))
    cards, has_more = load_cards(tab, config.FIRST_PAGE_SIZE)
    total = conn().execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    return render_template(
        "index.html", tabs=tabs(), tab=tab, cards=cards, has_more=has_more, total=total
    )


MAX_SEEN_IDS = 2000


@bp.get("/api/articles")
def api_articles():
    """HTML fragment with the next page of cards ("Load more").

    ?seen=1,2,3 lists the article ids already on the page.
    """
    tab = valid_tab(request.args.get("tab"))
    seen = {int(part) for part in request.args.get("seen", "").split(",")[:MAX_SEEN_IDS] if part.isdigit()}
    cards, has_more = load_cards(tab, config.PAGE_SIZE, seen)
    html = render_template("_cards.html", cards=cards, hero=False)
    return jsonify(html=html, has_more=has_more)


def feedback_kinds(database: sqlite3.Connection, article_id: int) -> set[str]:
    rows = database.execute("SELECT kind FROM feedback WHERE article_id = ?", (article_id,)).fetchall()
    return {row[0] for row in rows}


def record_feedback(database: sqlite3.Connection, article_id: int, kind: str, value: bool) -> bool:
    """Add (value=True) or remove one kind of feedback. Returns True if anything changed.

    up and down are mutually exclusive; an "open" cannot be taken back.
    """
    changes = database.total_changes
    if value:
        if kind in ("up", "down"):
            other = "down" if kind == "up" else "up"
            database.execute("DELETE FROM feedback WHERE article_id = ? AND kind = ?", (article_id, other))
        database.execute(
            "INSERT OR IGNORE INTO feedback (article_id, kind, created_at) VALUES (?, ?, ?)",
            (article_id, kind, db.now()),
        )
    elif kind != "open":
        database.execute("DELETE FROM feedback WHERE article_id = ? AND kind = ?", (article_id, kind))
    database.commit()
    return database.total_changes != changes


@bp.post("/api/feedback")
def api_feedback():
    """Record feedback and retrain.

    Body: {"article_id": 1, "kind": "up"|"down"|"save"|"open", "value": true|false}
    value=false removes that feedback (un-like, un-save). Only JSON is
    accepted, which also stops cross-site form posts.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="expected a JSON object"), 400
    kind, article_id, value = data.get("kind"), data.get("article_id"), data.get("value", True)
    if kind not in FEEDBACK_KINDS or not isinstance(article_id, int) or not isinstance(value, bool):
        return jsonify(error="invalid kind, article_id or value"), 400
    database = conn()
    if not database.execute("SELECT 1 FROM articles WHERE id = ?", (article_id,)).fetchone():
        return jsonify(error="no such article"), 404

    record_feedback(database, article_id, kind, value)
    started = time.perf_counter()
    model = rank.retrain(database)
    kinds = feedback_kinds(database, article_id)
    return jsonify(
        ok=True,
        article_id=article_id,
        up="up" in kinds,
        down="down" in kinds,
        saved="save" in kinds,
        n_labels=model.n_labels if model else 0,
        retrain_ms=round((time.perf_counter() - started) * 1000),
    )


# --- reader -----------------------------------------------------------------


def back_href() -> str:
    """Where the reader's Back link goes without JavaScript: the feed page we came from."""
    ref = urlsplit(request.referrer or "")
    if ref.netloc == request.host and ref.path == url_for("main.index"):
        return ref.path + (f"?{ref.query}" if ref.query else "")
    return url_for("main.index")


@bp.get("/read/<int:article_id>")
def read(article_id: int):
    """The in-app reader. Opening it counts as an "opened" event.

    Reader HTML is extracted on first open if the fetch job did not already
    store it (see core.reader for the states). The page is never cached by
    the browser, so the feedback buttons always show the current state.
    """
    database = conn()
    article = database.execute(
        "SELECT a.*, f.name AS source FROM articles a JOIN feeds f ON f.id = a.feed_id WHERE a.id = ?",
        (article_id,),
    ).fetchone()
    if article is None:
        abort(404)
    content = reader.ensure_reader_content(database, article)
    if record_feedback(database, article_id, "open", True):
        rank.retrain(database)
    kinds = feedback_kinds(database, article_id)
    body = reader.sanitize_html(content.html, article["url"]) if content.html else None
    response = make_response(
        render_template(
            "reader.html",
            article=article,
            content_html=body,
            up="up" in kinds,
            down="down" in kinds,
            saved="save" in kinds,
            back_href=back_href(),
            tabs=tabs(),
            tab=None,
        )
    )
    response.headers["Cache-Control"] = "no-store"
    return response


# --- sources ----------------------------------------------------------------


@bp.get("/sources")
def sources():
    feeds = conn().execute(
        "SELECT f.*, (SELECT COUNT(*) FROM articles a WHERE a.feed_id = f.id) AS item_count "
        "FROM feeds f WHERE f.removed = 0 ORDER BY f.default_topic, f.name COLLATE NOCASE"
    ).fetchall()
    topics = [t.name for t in config.load_topics()]
    return render_template(
        "sources.html", feeds=feeds, topics=topics, tabs=tabs(), tab=None, error=request.args.get("error")
    )


@bp.post("/sources/add")
def sources_add():
    url = request.form.get("url", "").strip()
    name = request.form.get("name", "").strip()
    topic = request.form.get("topic", "").strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return redirect(url_for("main.sources", error="Feed URL must start with http:// or https://"))
    if topic not in [t.name for t in config.load_topics()]:
        return redirect(url_for("main.sources", error="Pick a topic for the feed"))
    try:
        weight = min(2.0, max(0.1, float(request.form.get("weight", "1") or 1)))
    except ValueError:
        weight = 1.0
    database = conn()
    database.execute(
        "INSERT INTO feeds (url, name, default_topic, weight, created_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(url) DO UPDATE SET removed = 0, muted = 0, name = excluded.name, "
        "default_topic = excluded.default_topic, weight = excluded.weight",
        (url, name or parts.hostname, topic, weight, db.now()),
    )
    database.commit()
    return redirect(url_for("main.sources"))


@bp.post("/sources/<int:feed_id>/mute")
def sources_mute(feed_id: int):
    database = conn()
    database.execute("UPDATE feeds SET muted = 1 - muted WHERE id = ?", (feed_id,))
    database.commit()
    return redirect(url_for("main.sources"))


@bp.post("/sources/<int:feed_id>/remove")
def sources_remove(feed_id: int):
    """Hide a feed for good. Its articles disappear from the lists and expire normally."""
    database = conn()
    database.execute("UPDATE feeds SET removed = 1 WHERE id = ?", (feed_id,))
    database.commit()
    return redirect(url_for("main.sources"))


# --- images, PWA ------------------------------------------------------------


@bp.get("/img/<path:name>")
def image(name: str):
    response = send_from_directory(config.images_dir(), name)
    response.headers["Cache-Control"] = "public, max-age=2592000, immutable"
    return response


@bp.get("/manifest.webmanifest")
def manifest():
    return jsonify(
        name="Front Page",
        short_name="Front Page",
        description="Personal ML-ranked news",
        start_url="/",
        scope="/",
        display="standalone",
        background_color="#0d0f14",
        theme_color="#0d0f14",
        icons=[
            {"src": url_for("static", filename="icons/icon-192.png"), "sizes": "192x192", "type": "image/png"},
            {"src": url_for("static", filename="icons/icon-512.png"), "sizes": "512x512", "type": "image/png"},
            {
                "src": url_for("static", filename="icons/icon-maskable-512.png"),
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "maskable",
            },
        ],
    )


@bp.get("/sw.js")
def service_worker():
    """Served from the root so its scope covers the whole app."""
    body = render_template("sw.js", version=current_app.config["ASSET_VERSION"])
    response = make_response(body)
    response.headers["Content-Type"] = "application/javascript"
    response.headers["Cache-Control"] = "no-cache"
    return response


@bp.get("/offline")
def offline():
    return render_template("offline.html", tabs=tabs(), tab=None)


@bp.get("/healthz")
def healthz():
    conn().execute("SELECT 1").fetchone()
    return jsonify(ok=True)


# Defence in depth on top of the sanitiser: no inline or third-party script
# can run, nothing can be framed, and forms only post back to this app.
# Images may come from anywhere (inline article images that were not cached).
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "img-src 'self' https: http: data:",
        "connect-src 'self'",
        "frame-src 'none'",
        "frame-ancestors 'none'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
    ]
)


@bp.after_app_request
def security_headers(response):
    response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


@bp.app_errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify(error="not found"), 404
    return render_template("offline.html", tabs=tabs(), tab=None, message="Page not found."), 404

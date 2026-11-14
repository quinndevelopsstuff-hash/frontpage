"""Flask web app for Front Page. `app:create_app()` is the gunicorn entry point."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from flask import Flask, g

from core import colors, db


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(DB_PATH=None, MAX_CONTENT_LENGTH=64 * 1024)
    app.config.update(test_config or {})
    # Changes whenever a static file changes, so browsers and the service
    # worker pick up new CSS/JS after `git pull` without manual cache clearing.
    static_dir = Path(app.static_folder or "")
    app.config["ASSET_VERSION"] = str(
        int(max((p.stat().st_mtime for p in static_dir.rglob("*") if p.is_file()), default=time.time()))
    )

    def get_db() -> sqlite3.Connection:
        if "db" not in g:
            g.db = db.connect(app.config["DB_PATH"])
        return g.db

    @app.teardown_appcontext
    def close_db(_exc: BaseException | None) -> None:
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    app.extensions["get_db"] = get_db

    # Import scikit-learn now (about a second on a Pi) rather than during the
    # first feedback click, so retraining is fast from the first tap.
    import sklearn.linear_model  # noqa: F401

    from . import routes

    app.register_blueprint(routes.bp)
    app.jinja_env.filters["timeago"] = routes.timeago
    app.jinja_env.filters["fulldate"] = routes.fulldate
    app.jinja_env.filters["isodate"] = routes.isodate
    app.jinja_env.globals["topic_style"] = colors.topic_style  # from a topic's accent colour
    return app

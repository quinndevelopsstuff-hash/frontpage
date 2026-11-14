"""SQLite storage: connection, automatic migrations and small query helpers.

Migrations are a plain ordered list of SQL scripts. The schema_version table
holds a single integer; on connect we apply every script after that number,
each inside its own transaction, so `git pull && ./deploy/install.sh` (or just
restarting the app) upgrades the database safely. To change the schema, append
a new entry to MIGRATIONS -- never edit an existing one.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from . import config

MIGRATIONS: list[str] = [
    # 1: initial schema
    """
    CREATE TABLE feeds (
        id            INTEGER PRIMARY KEY,
        url           TEXT NOT NULL UNIQUE,
        name          TEXT NOT NULL,
        default_topic TEXT NOT NULL,
        weight        REAL NOT NULL DEFAULT 1.0,
        extract       INTEGER NOT NULL DEFAULT 1,
        max_items     INTEGER NOT NULL DEFAULT 40,
        muted         INTEGER NOT NULL DEFAULT 0,
        removed       INTEGER NOT NULL DEFAULT 0,
        etag          TEXT,
        last_modified TEXT,
        last_fetch_at REAL,
        last_ok_at    REAL,
        last_error    TEXT,
        created_at    REAL NOT NULL
    );

    CREATE TABLE articles (
        id           INTEGER PRIMARY KEY,
        feed_id      INTEGER NOT NULL REFERENCES feeds(id),
        url          TEXT NOT NULL UNIQUE,
        title        TEXT NOT NULL,
        author       TEXT,
        published_at REAL NOT NULL,
        fetched_at   REAL NOT NULL,
        summary      TEXT,
        content_text TEXT,
        image_url    TEXT,
        image_path   TEXT,
        topic        TEXT NOT NULL,
        embedding    BLOB,            -- float32, L2-normalised
        -- score cache, rewritten by core.rank.rescore_all
        score        REAL,            -- final 0..1 ranking score
        prior        REAL,            -- interest-similarity prior 0..1
        prior_label  TEXT,            -- which interest the prior matched
        model_score  REAL,            -- logistic regression probability 0..1
        reason       TEXT             -- the "why this?" line
    );
    CREATE INDEX idx_articles_score ON articles(score DESC);
    CREATE INDEX idx_articles_published ON articles(published_at);
    CREATE INDEX idx_articles_feed ON articles(feed_id);

    -- One row per (article, kind). kind is 'up', 'down', 'save' or 'open'.
    -- Toggling a button inserts or deletes the row.
    CREATE TABLE feedback (
        id         INTEGER PRIMARY KEY,
        article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
        kind       TEXT NOT NULL CHECK (kind IN ('up', 'down', 'save', 'open')),
        created_at REAL NOT NULL,
        UNIQUE (article_id, kind)
    );

    -- Small blobs: trained model weights, cached interest embeddings, etc.
    CREATE TABLE kv (
        key   TEXT PRIMARY KEY,
        value BLOB
    );
    """,
]


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the database (creating and migrating it if needed)."""
    conn = sqlite3.connect(str(path or config.db_path()), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    migrate(conn)
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        conn.commit()
        return 0
    return int(row[0])


def migrate(conn: sqlite3.Connection, migrations: list[str] | None = None) -> int:
    """Apply pending migrations. Returns the resulting schema version.

    Each migration runs in one transaction together with the version bump, so
    a crash half-way leaves the database at the previous version. BEGIN
    IMMEDIATE takes the write lock, so two processes starting at once (the two
    gunicorn workers) cannot both apply the same migration.
    """
    migrations = MIGRATIONS if migrations is None else migrations
    version = schema_version(conn)
    while version < len(migrations):
        old_isolation = conn.isolation_level
        conn.isolation_level = None  # manual transaction control
        try:
            conn.execute("BEGIN IMMEDIATE")
            current = int(conn.execute("SELECT version FROM schema_version").fetchone()[0])
            if current == version:
                for statement in _split_sql(migrations[version]):
                    conn.execute(statement)
                conn.execute("UPDATE schema_version SET version = ?", (version + 1,))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.isolation_level = old_isolation
        version = schema_version(conn)
    return version


def _split_sql(script: str) -> list[str]:
    """Split a migration into statements (no semicolons inside our DDL)."""
    lines = [line.split("--", 1)[0] for line in script.splitlines()]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def kv_get(conn: sqlite3.Connection, key: str) -> bytes | None:
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


def kv_set(conn: sqlite3.Connection, key: str, value: bytes | str) -> None:
    conn.execute(
        "INSERT INTO kv (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def now() -> float:
    return time.time()


if __name__ == "__main__":
    connection = connect()
    print(f"database at {config.db_path()} is at schema version {schema_version(connection)}")

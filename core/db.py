"""SQLite storage: connection, automatic migrations and small query helpers.

Migrations are an ordered list; each entry is a SQL script or a Python
function taking the connection (for data moves SQL alone can't express). The
schema_version table holds a single integer; on connect we apply every
migration after that number, each inside its own transaction, so
`git pull && ./deploy/install.sh` (or just restarting the app) upgrades the
database safely. To change the schema, append a new entry to MIGRATIONS --
never edit an existing one.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Callable, Union

from . import config

Migration = Union[str, Callable[[sqlite3.Connection], None]]


def _migrate_topics_table(conn: sqlite3.Connection) -> None:
    """3: topics move from config constants into a table.

    Creates `topics`, seeds it from config/interests.yaml, gives feeds and
    articles a topic_id (matching the old topic names case-insensitively;
    any name with no seed becomes a topic of its own so nothing is lost),
    then drops the old text columns.
    """
    from . import colors, topics  # local import: topics imports db

    if sqlite3.sqlite_version_info < (3, 35, 0):
        raise RuntimeError(f"SQLite 3.35+ is required (found {sqlite3.sqlite_version})")
    for statement in _split_sql(
        """
        CREATE TABLE topics (
            id            INTEGER PRIMARY KEY,
            name          TEXT NOT NULL,
            slug          TEXT NOT NULL UNIQUE,
            accent_color  TEXT NOT NULL,
            description   TEXT NOT NULL,     -- embedding anchor for assignment + prior
            position      INTEGER NOT NULL,  -- tab order
            created_at    REAL NOT NULL,
            seed_key      TEXT UNIQUE,       -- yaml topic name it was seeded from, if any
            interests     TEXT,              -- JSON {label: sentence} from the yaml
            user_modified INTEGER NOT NULL DEFAULT 0  -- 1 = created/edited in the UI; seeding leaves it alone
        );
        CREATE UNIQUE INDEX idx_topics_name ON topics(name COLLATE NOCASE);
        ALTER TABLE feeds ADD COLUMN topic_id INTEGER REFERENCES topics(id);
        ALTER TABLE articles ADD COLUMN topic_id INTEGER REFERENCES topics(id);
        """
    ):
        conn.execute(statement)
    topics.seed_topics(conn, commit=False)
    old_names = {r[0] for r in conn.execute("SELECT default_topic FROM feeds UNION SELECT topic FROM articles")}
    for name in sorted(n for n in old_names if n):
        if topics.id_by_name(conn, name) is None:
            position = conn.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM topics").fetchone()[0]
            used = {r[0] for r in conn.execute("SELECT accent_color FROM topics")}
            accent = next((h for h in colors.PRESET_HEXES if h not in used), colors.PRESET_HEXES[0])
            conn.execute(
                "INSERT INTO topics (name, slug, accent_color, description, position, created_at, user_modified) "
                "VALUES (?, ?, ?, ?, ?, ?, 1)",
                (name, topics._unique_slug(conn, name) or f"topic-{position}", accent, name, position, now()),
            )
    for table, column in (("feeds", "default_topic"), ("articles", "topic")):
        for (name,) in conn.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall():
            conn.execute(f"UPDATE {table} SET topic_id = ? WHERE {column} = ?", (topics.id_by_name(conn, name), name))
        conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    conn.execute("CREATE INDEX idx_articles_topic ON articles(topic_id)")
    conn.execute("CREATE INDEX idx_feeds_topic ON feeds(topic_id)")
    # Topic anchors are now cached per sentence; the old blob is obsolete.
    conn.execute("DELETE FROM kv WHERE key LIKE 'interests:%'")


MIGRATIONS: list[Migration] = [
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
    # 2: reader view (see core/reader.py for the states)
    """
    ALTER TABLE articles ADD COLUMN content_html TEXT;
    ALTER TABLE articles ADD COLUMN reader_state TEXT;
    ALTER TABLE articles ADD COLUMN reader_checked_at REAL;
    """,
    # 3: dynamic topics
    _migrate_topics_table,
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


def migrate(conn: sqlite3.Connection, migrations: list[Migration] | None = None) -> int:
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
                step = migrations[version]
                if callable(step):
                    step(conn)
                else:
                    for statement in _split_sql(step):
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

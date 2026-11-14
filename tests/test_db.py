import sqlite3

import pytest

from core import db


def test_fresh_database_is_migrated_to_latest(tmp_path):
    conn = db.connect(tmp_path / "new.db")
    assert db.schema_version(conn) == len(db.MIGRATIONS)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"feeds", "articles", "feedback", "kv", "schema_version"} <= tables


def test_connect_twice_is_a_no_op(tmp_path):
    path = tmp_path / "twice.db"
    first = db.connect(path)
    first.execute(
        "INSERT INTO feeds (url, name, created_at) VALUES ('https://example.com/f', 'F', 0)"
    )
    first.commit()
    second = db.connect(path)
    assert db.schema_version(second) == len(db.MIGRATIONS)
    assert second.execute("SELECT COUNT(*) FROM feeds").fetchone()[0] == 1


def test_new_migration_upgrades_existing_data(tmp_path):
    path = tmp_path / "upgrade.db"
    conn = db.connect(path)
    conn.execute(
        "INSERT INTO feeds (url, name, created_at) VALUES ('https://example.com/f', 'F', 0)"
    )
    conn.commit()
    future = db.MIGRATIONS + ["ALTER TABLE feeds ADD COLUMN note TEXT DEFAULT 'hi'; CREATE TABLE extra (id INTEGER);"]
    assert db.migrate(conn, future) == len(db.MIGRATIONS) + 1
    assert conn.execute("SELECT name, note FROM feeds").fetchone()[:] == ("F", "hi")
    # running again changes nothing
    assert db.migrate(conn, future) == len(db.MIGRATIONS) + 1


def test_failed_migration_rolls_back_completely(tmp_path):
    conn = db.connect(tmp_path / "bad.db")
    before = db.schema_version(conn)
    broken = db.MIGRATIONS + ["CREATE TABLE half_done (id INTEGER); ALTER TABLE no_such_table ADD COLUMN x TEXT;"]
    with pytest.raises(sqlite3.OperationalError):
        db.migrate(conn, broken)
    assert db.schema_version(conn) == before
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "half_done" not in tables


def test_kv_roundtrip(conn):
    assert db.kv_get(conn, "missing") is None
    db.kv_set(conn, "k", b"one")
    db.kv_set(conn, "k", b"two")
    assert db.kv_get(conn, "k") == b"two"

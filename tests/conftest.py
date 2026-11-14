"""Shared fixtures: an isolated data dir, the offline embedder, a Flask client."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db, embed  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Every test gets its own data/ directory and the hash embedder."""
    monkeypatch.setenv("FRONTPAGE_DATA", str(tmp_path / "data"))
    embed.set_embedder(embed.HashEmbedder())
    yield
    embed.set_embedder(None)


@pytest.fixture
def conn():
    connection = db.connect()
    yield connection
    connection.close()


@pytest.fixture
def seeded(conn):
    """A database holding the demo articles, embedded and scored."""
    from core import seed_demo

    seed_demo.seed(conn)
    return conn


@pytest.fixture
def client(seeded):
    from app import create_app

    app = create_app({"TESTING": True})
    return app.test_client()

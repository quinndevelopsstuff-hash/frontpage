"""Topics: the `topics` table, seeding from config/interests.yaml, and edits.

A topic is a row with a name, a URL slug, an accent colour, a description
and a position (tab order). The description is the topic's embedding anchor:
core.rank compares articles with it for topic assignment and for the
cold-start prior. Seeded topics can also carry finer "interests" from the
yaml file (e.g. "F1 engines"), which give the prior more specific anchors.

Seeding rules (seed_topics, run on migration and at the start of every
fetch run):
  * a yaml topic not in the table is inserted, unless you deleted it in the
    UI (it is then remembered in kv 'topics:deleted_seeds');
  * a yaml topic you have not touched in the UI is updated from the yaml;
  * a topic you created, renamed, recoloured or re-described in the UI
    (user_modified = 1) is never overwritten.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from . import colors, config, db

RESERVED_SLUGS = {"for-you", "saved"}  # the two fixed tabs
MAX_NAME = 30
MAX_DESCRIPTION = 400
DELETED_SEEDS_KEY = "topics:deleted_seeds"
RECLASSIFY_PENDING_KEY = "topics:reclassify_pending"  # set when the web app could not embed yet


class TopicError(ValueError):
    """A topic edit was rejected; the message is shown to the user."""


@dataclass(frozen=True)
class Topic:
    id: int
    name: str
    slug: str
    accent_color: str
    description: str
    position: int
    interests: dict[str, str]

    @property
    def style(self) -> dict[str, str]:
        return colors.topic_style(self.accent_color)


def _from_row(row: sqlite3.Row) -> Topic:
    return Topic(
        id=row["id"],
        name=row["name"],
        slug=row["slug"],
        accent_color=row["accent_color"],
        description=row["description"],
        position=row["position"],
        interests=json.loads(row["interests"] or "{}"),
    )


def list_topics(conn: sqlite3.Connection) -> list[Topic]:
    """All topics in tab order."""
    return [_from_row(r) for r in conn.execute("SELECT * FROM topics ORDER BY position, id")]


def get(conn: sqlite3.Connection, topic_id: int) -> Topic | None:
    row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
    return _from_row(row) if row else None


def by_slug(conn: sqlite3.Connection, slug: str) -> Topic | None:
    row = conn.execute("SELECT * FROM topics WHERE slug = ?", (slug,)).fetchone()
    return _from_row(row) if row else None


def id_by_name(conn: sqlite3.Connection, name: str) -> int | None:
    """Case-insensitive lookup by name (or slug), e.g. for feeds.yaml."""
    wanted = name.strip().casefold()
    for row in conn.execute("SELECT id, name, slug FROM topics"):
        if row["name"].casefold() == wanted or row["slug"] == wanted:
            return row["id"]
    return None


# --- validation -------------------------------------------------------------


def slugify(name: str) -> str:
    """'Formula 1 & More' -> 'formula-1-more'."""
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")


def _unique_slug(conn: sqlite3.Connection, name: str, exclude_id: int | None = None) -> str:
    base = slugify(name)
    slug, n = base, 2
    while conn.execute("SELECT 1 FROM topics WHERE slug = ? AND id IS NOT ?", (slug, exclude_id)).fetchone():
        slug, n = f"{base}-{n}", n + 1
    return slug


def validate_name(conn: sqlite3.Connection, name: str, exclude_id: int | None = None) -> str:
    """Trimmed name, or TopicError: empty, too long, reserved or duplicate (any case)."""
    name = " ".join((name or "").split())
    if not name:
        raise TopicError("Give the topic a name.")
    if len(name) > MAX_NAME:
        raise TopicError(f"Keep the name under {MAX_NAME} characters.")
    slug = slugify(name)
    if not slug:
        raise TopicError("The name needs at least one letter or digit.")
    if slug in RESERVED_SLUGS:
        raise TopicError(f'"{name}" is already used by a built-in tab.')
    for row in conn.execute("SELECT id, name FROM topics"):
        if row["id"] != exclude_id and row["name"].casefold() == name.casefold():
            raise TopicError(f'There is already a topic called "{row["name"]}".')
    return name


def validate_description(description: str) -> str:
    description = " ".join((description or "").split())
    if not description:
        raise TopicError("Describe the topic in a sentence; it is how articles get matched to it.")
    if len(description) > MAX_DESCRIPTION:
        raise TopicError(f"Keep the description under {MAX_DESCRIPTION} characters.")
    return description


def validate_accent(accent: str, current: str | None = None) -> str:
    """One of the preset swatches (or the topic's current colour, unchanged)."""
    accent = colors.normalize_hex(accent or "")
    if accent in colors.PRESET_HEXES or (current and accent == colors.normalize_hex(current)):
        return accent
    raise TopicError("Pick one of the colour swatches.")


# --- edits from the UI --------------------------------------------------------


def create_topic(conn: sqlite3.Connection, name: str, description: str, accent: str) -> Topic:
    """Validate and insert a user topic at the end of the tab bar."""
    name = validate_name(conn, name)
    description = validate_description(description)
    accent = validate_accent(accent)
    position = conn.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM topics").fetchone()[0]
    cur = conn.execute(
        "INSERT INTO topics (name, slug, accent_color, description, position, created_at, user_modified) "
        "VALUES (?, ?, ?, ?, ?, ?, 1)",
        (name, _unique_slug(conn, name), accent, description, position, db.now()),
    )
    conn.commit()
    return get(conn, cur.lastrowid)  # type: ignore[return-value]


def update_topic(conn: sqlite3.Connection, topic_id: int, name: str, description: str, accent: str) -> Topic:
    """Rename / re-describe / recolour. Marks the topic as user-owned."""
    topic = get(conn, topic_id)
    if topic is None:
        raise TopicError("That topic no longer exists.")
    name = validate_name(conn, name, exclude_id=topic_id)
    description = validate_description(description)
    accent = validate_accent(accent, current=topic.accent_color)
    slug = topic.slug if name == topic.name else _unique_slug(conn, name, exclude_id=topic_id)
    conn.execute(
        "UPDATE topics SET name = ?, slug = ?, description = ?, accent_color = ?, user_modified = 1 WHERE id = ?",
        (name, slug, description, accent, topic_id),
    )
    conn.commit()
    return get(conn, topic_id)  # type: ignore[return-value]


def feed_count(conn: sqlite3.Connection, topic_id: int) -> int:
    return conn.execute("SELECT COUNT(*) FROM feeds WHERE topic_id = ?", (topic_id,)).fetchone()[0]


def delete_topic(conn: sqlite3.Connection, topic_id: int, move_feeds_to: int | None) -> None:
    """Delete a topic, moving its feeds (and articles) elsewhere.

    Feeds that use the topic need a destination; their articles follow them.
    Articles filed under the topic by similarity go back to their own feed's
    topic. The last remaining topic cannot be deleted.
    """
    topic = get(conn, topic_id)
    if topic is None:
        raise TopicError("That topic no longer exists.")
    if conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0] <= 1:
        raise TopicError("Front Page needs at least one topic.")
    if feed_count(conn, topic_id):
        if move_feeds_to is None or move_feeds_to == topic_id or get(conn, move_feeds_to) is None:
            raise TopicError(f'Choose where the feeds in "{topic.name}" should go.')
        conn.execute("UPDATE feeds SET topic_id = ? WHERE topic_id = ?", (move_feeds_to, topic_id))
    conn.execute(
        "UPDATE articles SET topic_id = (SELECT f.topic_id FROM feeds f WHERE f.id = articles.feed_id) "
        "WHERE topic_id = ?",
        (topic_id,),
    )
    seed_key = conn.execute("SELECT seed_key FROM topics WHERE id = ?", (topic_id,)).fetchone()[0]
    if seed_key:  # don't let the next fetch run re-create it from interests.yaml
        deleted = set(json.loads(db.kv_get(conn, DELETED_SEEDS_KEY) or "[]"))
        db.kv_set(conn, DELETED_SEEDS_KEY, json.dumps(sorted(deleted | {seed_key})))
    conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
    conn.commit()


# --- seeding from interests.yaml ----------------------------------------------


def seed_topics(conn: sqlite3.Connection, seeds: list[config.TopicSeed] | None = None, commit: bool = True) -> int:
    """Apply config/interests.yaml to the topics table. Returns rows inserted.

    commit=False lets a migration run this inside its own transaction.
    """
    seeds = config.load_topics() if seeds is None else seeds
    deleted = set(json.loads(db.kv_get(conn, DELETED_SEEDS_KEY) or "[]"))
    inserted = 0
    for seed in seeds:
        key = seed.name.casefold()
        accent = colors.normalize_hex(seed.accent) if colors.is_hex(seed.accent) else _spare_color(conn)
        interests = json.dumps(seed.interests)
        row = conn.execute("SELECT id, user_modified FROM topics WHERE seed_key = ?", (key,)).fetchone()
        if row is not None:
            if not row["user_modified"]:
                conn.execute(
                    "UPDATE topics SET description = ?, accent_color = ?, interests = ? WHERE id = ?",
                    (seed.description, accent, interests, row["id"]),
                )
            continue
        if key in deleted:
            continue
        existing = id_by_name(conn, seed.name)
        if existing is not None:  # you made a topic with this name yourself; adopt it, keep your edits
            conn.execute("UPDATE topics SET seed_key = ? WHERE id = ? AND seed_key IS NULL", (key, existing))
            continue
        position = conn.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM topics").fetchone()[0]
        conn.execute(
            "INSERT INTO topics (name, slug, accent_color, description, position, created_at, seed_key, interests) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (seed.name, _unique_slug(conn, seed.name), accent, seed.description, position, db.now(), key, interests),
        )
        inserted += 1
    if commit:
        conn.commit()
    return inserted


def _spare_color(conn: sqlite3.Connection) -> str:
    """The first preset no topic uses yet (or the first preset)."""
    used = {row[0].upper() for row in conn.execute("SELECT accent_color FROM topics")}
    return next((h for h in colors.PRESET_HEXES if h not in used), colors.PRESET_HEXES[0])


def suggested_accent(conn: sqlite3.Connection) -> str:
    return _spare_color(conn)

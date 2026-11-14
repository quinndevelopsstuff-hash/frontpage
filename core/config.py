"""Paths, tunables and YAML config loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

USER_AGENT = "FrontPage/1.0 (personal self-hosted RSS reader)"
HTTP_TIMEOUT = 10  # seconds, for every outbound request
MAX_FEED_BYTES = 5 * 1024 * 1024
MAX_PAGE_BYTES = 3 * 1024 * 1024
MAX_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_WIDTH = 800
MAX_TEXT_CHARS = 20_000
MAX_SUMMARY_CHARS = 600
RETENTION_DAYS = 30
DEFAULT_MAX_ITEMS = 40  # newest entries considered per feed per run
MAX_PAGE_FETCHES_PER_RUN = 80  # article pages downloaded per run (gentle)
PAGE_FETCH_PAUSE = 0.5  # seconds between article page downloads
DEMO_URL_PREFIX = "https://example.com/frontpage-demo/"  # core.seed_demo articles
FIRST_PAGE_SIZE = 20  # hero + side card + 18 fills the 3-column grid evenly
PAGE_SIZE = 18  # each "Load more"

# Accent colour and typographic-cover gradient per topic (from the mockups).
TOPIC_STYLES: dict[str, dict[str, str]] = {
    "F1": {"accent": "#FF6B6B", "tint": "#F7C6C6", "grad": "#7a1620, #3b0f1c 60%, #14101a"},
    "AI": {"accent": "#A78BFA", "tint": "#DAD0FB", "grad": "#3b2a7a, #1d1840 60%, #12121f"},
    "CS": {"accent": "#60A5FA", "tint": "#C5DDFB", "grad": "#0f3d73, #0c2342 60%, #0d1220"},
    "Engineering": {"accent": "#FBBF24", "tint": "#F3E3B0", "grad": "#5c4208, #2e2208 70%, #17140c"},
}
DEFAULT_TOPIC_STYLE = {"accent": "#9AA3B2", "tint": "#D5DAE2", "grad": "#2a3040, #1b202b 60%, #12151c"}


def data_dir() -> Path:
    """Runtime data directory (db, images, model cache). Override with FRONTPAGE_DATA."""
    path = Path(os.environ.get("FRONTPAGE_DATA", ROOT / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return data_dir() / "frontpage.db"


def images_dir() -> Path:
    path = data_dir() / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class FeedSeed:
    url: str
    name: str
    default_topic: str
    weight: float = 1.0
    extract: bool = True
    max_items: int = DEFAULT_MAX_ITEMS


@dataclass(frozen=True)
class Topic:
    name: str
    about: str
    interests: dict[str, str] = field(default_factory=dict)  # label -> description


def load_feed_seeds(path: Path | None = None) -> list[FeedSeed]:
    """Read config/feeds.yaml."""
    raw = yaml.safe_load((path or CONFIG_DIR / "feeds.yaml").read_text()) or {}
    seeds = []
    for item in raw.get("feeds", []):
        seeds.append(
            FeedSeed(
                url=str(item["url"]).strip(),
                name=str(item.get("name") or item["url"]),
                default_topic=str(item.get("default_topic", "CS")),
                weight=float(item.get("weight", 1.0)),
                extract=bool(item.get("extract", True)),
                max_items=int(item.get("max_items", DEFAULT_MAX_ITEMS)),
            )
        )
    return seeds


def load_topics(path: Path | None = None) -> list[Topic]:
    """Read config/interests.yaml. Order in the file is the tab order."""
    raw = yaml.safe_load((path or CONFIG_DIR / "interests.yaml").read_text()) or {}
    topics = []
    for name, body in (raw.get("topics") or {}).items():
        body = body or {}
        interests = {str(k): str(v) for k, v in (body.get("interests") or {}).items()}
        topics.append(Topic(name=str(name), about=str(body.get("about", name)), interests=interests))
    return topics


def topic_style(topic: str) -> dict[str, str]:
    return TOPIC_STYLES.get(topic, DEFAULT_TOPIC_STYLE)

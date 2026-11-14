"""`python -m core.seed_demo`: load a few fake articles to look at the UI offline.

Demo articles live under example.com URLs in a feed called "Demo" per topic.
Run with --clear to remove them again. If the real embedding model is not
available (no network yet), a simple offline embedder is used instead; the
next real fetch run re-embeds everything with the real model.
"""
from __future__ import annotations

import argparse

from . import db, embed, fetch
from .config import DEMO_URL_PREFIX

# (topic, source name, hours ago, title, summary)
DEMO_ARTICLES: list[tuple[str, str, float, str, str]] = [
    ("F1", "Demo: The Race", 2, "How a rule change could reshape next season's engine race",
     "Formula 1 power unit regulations are shifting again, and the manufacturers who read the new hybrid and fuel rules best could start with a real advantage."),
    ("AI", "Demo: arXiv cs.LG", 5, "A smaller model that matches a larger one by reading its training data differently",
     "We show that curriculum ordering and distillation let a compact transformer language model reach the benchmark accuracy of one four times its size."),
    ("CS", "Demo: Hacker News", 3, "Why a decades-old scheduling idea keeps coming back",
     "Operating systems and distributed databases keep rediscovering the same scheduler design. A look at why it performs so well under load."),
    ("F1", "Demo: RaceFans", 6, "What the data says about tyre wear across the last three rounds",
     "Grand prix lap time and pit stop data show which teams manage tyre strategy best over a race distance."),
    ("Engineering", "Demo: Hackaday", 8, "Building a motor controller from scratch",
     "A field-oriented brushless motor controller built around a cheap microcontroller, with the PCB design and firmware explained step by step."),
    ("AI", "Demo: arXiv cs.AI", 11, "Agents that plan before they call a tool make fewer mistakes",
     "An evaluation of AI agents on multi-step reasoning tasks finds that explicit planning cuts tool use errors substantially."),
    ("CS", "Demo: Ars Technica", 14, "A new compiler pass finds bugs that fuzzers miss",
     "Researchers describe a static analysis that catches memory safety vulnerabilities in open source systems software."),
    ("Engineering", "Demo: IEEE Spectrum", 20, "The battery pack that charges in ten minutes",
     "Power electronics and cell chemistry advances are bringing ten-minute electric vehicle charging closer to production."),
    ("F1", "Demo: Autosport", 26, "Inside the floor upgrade that turned a midfield car around",
     "A technical analysis of the aerodynamics development and wind tunnel work behind a mid-season Formula 1 upgrade."),
    ("CS", "Demo: Hacker News", 30, "Running a home server on a Raspberry Pi for five years",
     "Lessons from self-hosting on Linux with a Raspberry Pi: what broke, what lasted, and the software worth running."),
    ("AI", "Demo: arXiv cs.LG", 34, "Quantizing language models to four bits without losing accuracy",
     "A quantization method for efficient machine learning inference on edge devices keeps accuracy within a point of full precision."),
    ("Engineering", "Demo: Hackaday", 40, "A teardown of a twenty-dollar thermal camera",
     "Hardware hacking a cheap thermal sensor module reveals a surprisingly capable design and an undocumented serial port."),
    ("AI", "Demo: Ars Technica", 46, "What interpretability research can and cannot tell us",
     "Researchers studying AI safety map the circuits inside a neural network, and explain the limits of the approach."),
    ("Engineering", "Demo: IEEE Spectrum", 52, "The reusable rocket engine that survived forty flights",
     "Aerospace engineers describe the materials and inspection methods that keep a spacecraft engine flying."),
    ("F1", "Demo: The Race", 60, "The driver market move nobody saw coming",
     "Formula 1 team politics and contract clauses explain a surprise switch ahead of the championship run-in."),
    ("CS", "Demo: Ars Technica", 70, "A database that treats time as a first-class type",
     "A new open source database engine rethinks storage and query performance for time series workloads."),
    ("Engineering", "Demo: Hackaday", 80, "Printing gears that actually last",
     "3D printing tough, quiet gears for robotics builds takes the right material, tooth profile and a little post-processing."),
    ("AI", "Demo: arXiv cs.AI", 90, "Reinforcement learning agents learn to cooperate from sparse rewards",
     "A reinforcement learning study of multi-agent planning shows cooperation emerging without reward shaping."),
    ("F1", "Demo: RaceFans", 100, "Qualifying pace versus race pace: who is hiding what",
     "Sector times and fuel-corrected long runs suggest the grand prix pecking order differs from qualifying."),
    ("CS", "Demo: Hacker News", 110, "The cryptography bug that hid in plain sight for a decade",
     "A subtle vulnerability in a widely used security library shows how hard constant-time code is to get right."),
    ("Engineering", "Demo: IEEE Spectrum", 125, "Why grid-scale storage is moving beyond lithium",
     "Energy systems engineers weigh iron-air and sodium batteries for long-duration storage."),
    ("AI", "Demo: arXiv cs.LG", 140, "Fine-tuning with a thousand examples: what really matters",
     "A benchmark of fine-tuning methods for transformers finds data quality beats every training trick."),
    ("CS", "Demo: Ars Technica", 160, "A programming language designed around error messages",
     "The designers of a new programming language explain why developer tools came before syntax."),
    ("F1", "Demo: Autosport", 180, "How the FIA plans to police flexible wings",
     "New load tests and rule changes target wing flex, and teams are already arguing about the wording."),
]


# Articles whose reader view shows the "Open original" fallback in the demo.
DEMO_FALLBACK_INDEXES = {len(DEMO_ARTICLES) - 1}


def demo_body(title: str, summary: str) -> str:
    """Placeholder reader HTML that exercises every element the reader styles."""
    return (
        f"<p>{summary}</p>"
        "<p>This is demo text so you can see the reader view without a network connection. "
        "Real articles show the page's own text here, extracted when the feed is fetched or the "
        "first time you open them, and cleaned so that no scripts or embeds come along.</p>"
        "<h2>What the reader keeps</h2>"
        "<p>Paragraphs, headings, lists, quotes, <a href=\"https://example.com/\">links</a>, "
        "<strong>bold</strong> and <em>italic</em> text, <code>inline code</code> and images.</p>"
        "<ul><li>Links open the original site in a new tab.</li>"
        "<li>Images are copied into the local image cache.</li>"
        "<li>Thumbs and save work exactly as they do on the cards.</li></ul>"
        f"<blockquote>“{title}” is only a sample headline.</blockquote>"
        "<p>Use Back to return to the feed at the same place you left it.</p>"
    )


def clear(conn) -> int:
    """Remove demo articles and the demo feeds."""
    like = DEMO_URL_PREFIX + "%"
    count = conn.execute("SELECT COUNT(*) FROM articles WHERE url LIKE ?", (like,)).fetchone()[0]
    conn.execute("DELETE FROM articles WHERE url LIKE ?", (like,))
    conn.execute("DELETE FROM feeds WHERE url LIKE ?", (like,))
    conn.commit()
    return count


def seed(conn) -> int:
    """Insert the demo articles (idempotent) and score them."""
    now = db.now()
    for i, (topic, source, hours, title, summary) in enumerate(DEMO_ARTICLES):
        feed_url = f"{DEMO_URL_PREFIX}feed/{source.replace(' ', '-').replace(':', '').lower()}"
        conn.execute(
            "INSERT OR IGNORE INTO feeds (url, name, default_topic, weight, created_at) VALUES (?, ?, ?, 1.0, ?)",
            (feed_url, source, topic, now),
        )
        feed_id = conn.execute("SELECT id FROM feeds WHERE url = ?", (feed_url,)).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO articles (feed_id, url, title, author, published_at, fetched_at, summary, "
            "content_text, topic, content_html, reader_state, reader_checked_at) "
            "VALUES (?, ?, ?, 'Demo Author', ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                feed_id,
                f"{DEMO_URL_PREFIX}{i}",
                title,
                now - hours * 3600,
                now,
                summary,
                summary,
                topic,
                None if i in DEMO_FALLBACK_INDEXES else demo_body(title, summary),
                "failed" if i in DEMO_FALLBACK_INDEXES else "ready",
                now,
            ),
        )
    conn.commit()
    fetch.index_and_score(conn, _embedder(conn))
    return len(DEMO_ARTICLES)


def _embedder(conn) -> embed.Embedder:
    """The real model if it loads; the offline fallback only for a demo-only database.

    (Falling back with real articles present would throw their embeddings away.)
    """
    real = conn.execute("SELECT COUNT(*) FROM articles WHERE url NOT LIKE ?", (DEMO_URL_PREFIX + "%",)).fetchone()[0]
    return embed.get_embedder(allow_fallback=real == 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load or remove demo articles.")
    parser.add_argument("--clear", action="store_true", help="remove the demo articles")
    args = parser.parse_args()
    conn = db.connect()
    if args.clear:
        print(f"removed {clear(conn)} demo articles")
    else:
        count = seed(conn)
        print(f"seeded {count} demo articles using the '{embed.get_embedder().name}' embedder")
        print("start the app and open http://127.0.0.1:9092  (remove them with: python -m core.seed_demo --clear)")


if __name__ == "__main__":
    main()

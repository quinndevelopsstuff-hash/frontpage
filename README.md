# Front Page

A personal, self-hosted news front page for a Raspberry Pi 5. It fetches RSS
feeds, ranks every article by how likely *you* are to care, and learns from
your thumbs-up / thumbs-down / save / open feedback. No Docker, no cloud, no
npm; one install script.

Phase 1 covers fetching, ranking, the web UI, the Sources page and deployment.

## How it works

```
                 every 20 min (systemd timer, low priority)
 config/feeds.yaml ──seed──▶ ┌───────────────────────────────────────────────┐
 config/interests.yaml ────▶ │ python -m core.fetch                          │
                             │  fetch feeds (ETag / Last-Modified)           │
                             │  → dedupe (canonical URL, near-same title)    │
                             │  → article text (trafilatura) + image (Pillow)│
                             │  → embed (all-MiniLM-L6-v2, ONNX)             │
                             │  → prior + topic → score → 30-day retention   │
                             └──────────────────────┬────────────────────────┘
                                                    ▼
                                     data/frontpage.db  (SQLite, WAL)
                                     data/images/  data/models/
                                                    ▲
                             ┌──────────────────────┴────────────────────────┐
  phone / laptop ──HTTPS──▶  │ gunicorn (2×4 threads) 127.0.0.1:9092         │
  via `tailscale serve`      │  Flask + Jinja + vanilla JS (app/), reader    │
                             │  feedback → retrain logistic regression →     │
                             │  rescore all articles (milliseconds)          │
                             └───────────────────────────────────────────────┘
```

| Path | What |
| --- | --- |
| `core/fetch.py` | the fetch job (`python -m core.fetch`) |
| `core/dedupe.py` | URL canonicalisation, near-duplicate titles |
| `core/net.py` | HTTP with SSRF guard, size cap, timeout |
| `core/extract.py` | article text, og:image, image cache |
| `core/reader.py` | reader view: HTML extraction, sanitiser (nh3), inline image caching |
| `core/embed.py` | embeddings (fastembed / ONNX) |
| `core/rank.py` | prior, model, blend, explore, "why this?" (heavily commented; start here) |
| `core/db.py` | SQLite schema and automatic migrations |
| `app/` | Flask routes, templates, CSS/JS, PWA |
| `deploy/` | `install.sh` and systemd units |

### Reader view

Tapping a card opens `/read/<id>` inside the app, which also records the
"opened" signal. The article body is trafilatura's HTML output (paragraphs,
headings, lists, quotes, links, images), cleaned by an allowlist sanitiser
(nh3) so no scripts, iframes, styles or event handlers survive, and sanitised
again on every render. When the fetch job already downloaded the page (or the
feed carries the full text) the HTML is stored then; otherwise it is extracted
the first time you open the article. Inline images are copied into the local
image cache on first open (up to 12, within 8 seconds; the rest load from the
original site without a referrer). If extraction fails or yields under ~300
characters, the reader shows the summary and an "Open original" button. Back
returns to the same place in the feed, including cards added with "Load more".

### Ranking in one paragraph

Each article is embedded (title + first ~500 words). The **prior** is its best
cosine similarity to the interest lines in `config/interests.yaml`. The
**model** is a logistic regression trained on your feedback (👍 1.0, 👎 1.0,
saved 1.5, opened 0.3; ignored articles are not negatives). They are blended:
`final = a·prior + (1−a)·model`, `a = max(0, 1 − n_labels/50)`, then multiplied
by a mild recency decay and the feed's weight. About one position in ten is an
**explore** slot from lower-scored articles in topics you've rated least. Every
card says why it is there ("matches your interest …" while the prior dominates,
"similar to … you saved" once the model does). The model retrains on every
feedback click.

## Install on the Pi

Needs 64-bit Raspberry Pi OS (Bookworm or newer, Python 3.11+) and `git`.

```bash
git clone <this repo> ~/frontpage
```

```bash
cd ~/frontpage && ./deploy/install.sh
```

The script is idempotent. It creates `.venv`, installs `requirements.txt`
(prebuilt aarch64 wheels, nothing compiles), creates `data/`, migrates the
database, installs and starts `frontpage.service` (web) and
`frontpage-fetch.timer` (every 20 minutes), and starts a first fetch in the
background. The first fetch downloads the embedding model (~90 MB) into
`data/models/` and takes a few minutes; the page shows "No articles yet"
until it finishes.

First thing to do after installing: confirm every feed URL answers.

```bash
.venv/bin/python -m core.fetch --check
```

### Expose it on your tailnet

The app listens on `127.0.0.1:9092` only. Publish it with Tailscale Serve
(HTTPS is required for "Add to Home Screen" to work as an app):

```bash
sudo tailscale serve --bg --https=9092 http://127.0.0.1:9092
```

That gives `https://<pi-name>.<tailnet>.ts.net:9092`, alongside anything else
you serve. To make it the default site on port 443 instead:

```bash
sudo tailscale serve --bg 9092
```

Check or undo with `tailscale serve status` and `tailscale serve reset`.
There is no login: anyone on your tailnet who can reach the Pi can use it.

On the iPhone: open the URL in Safari → Share → Add to Home Screen.

### Updating

```bash
cd ~/frontpage && git pull && ./deploy/install.sh
```

Database migrations run automatically (a `schema_version` table; each
migration is one transaction, so a failed upgrade leaves the old schema).

## Day to day

| I want to… | Do this |
| --- | --- |
| add / mute / remove a feed | the Sources page (menu button, top right) |
| change a feed's name, topic or weight | edit `config/feeds.yaml`; applied on the next fetch |
| change what I'm interested in | edit `config/interests.yaml`; priors are recomputed on the next fetch |
| fetch right now | `sudo systemctl start frontpage-fetch` |
| see web logs | `journalctl -u frontpage -f` |
| see fetch logs | `journalctl -u frontpage-fetch -f` |
| see when the timer fires next | `systemctl list-timers frontpage-fetch` |
| back up | copy `data/frontpage.db` (everything else can be rebuilt) |

`feeds.yaml` is a *seed*: feeds in it are added to the database, and edits to
name/topic/weight are applied, but a feed you removed or muted in the UI stays
that way. Feed errors (timeouts, 404s, not-a-feed) are logged and shown on the
Sources page; one broken feed never stops the others.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m pytest
```

See the UI without any network (24 fake articles; `--clear` removes them):

```bash
.venv/bin/python -m core.seed_demo
```

```bash
.venv/bin/gunicorn --bind 127.0.0.1:9092 "app:create_app()"
```

If the embedding model has not been downloaded yet, the demo uses a simple
offline word-hashing embedder; the next real fetch re-embeds everything with
the real model. `FRONTPAGE_DATA=/some/dir` moves the data directory.

## Design notes

- **Security.** Every URL from a feed or page goes through `core/net.py`:
  http(s) only, hosts that resolve to private, loopback, link-local or tailnet
  addresses are refused on every redirect hop, bodies are capped (images 5 MB)
  and time out after 10 s. The feedback API only accepts JSON, which blocks
  cross-site form posts. Reader HTML is allowlist-sanitised, and every page
  is served with a Content-Security-Policy that forbids inline and
  third-party scripts and framing.
- **Disk.** Articles expire after 30 days unless saved or rated; each fetch
  run also deletes cached images no remaining article uses.
- **Gentle on the Pi.** The fetch job is sequential, pauses between page
  downloads, downloads at most 80 article pages per run, uses 2 ONNX threads,
  and runs at `Nice=15` with idle IO priority. The web workers never load the
  embedding model.
- **Not built yet (Phase 2+).** Daily digest, topic clustering, taste stats,
  push notifications, multi-user. Feedback is stored as raw
  events and embeddings are kept per article, so these can be added without
  reshaping the data.

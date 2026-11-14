"""Ranking: how Front Page decides what you see first.

The whole pipeline, in the order it runs
----------------------------------------
Every article has a 384-number embedding (core.embed). From it we compute:

1. PRIOR  (cold start, no feedback needed)
   Each line of config/interests.yaml is embedded too. The prior is the
   article's best cosine similarity to any of those lines, rescaled to 0..1.
   "Does this look like something I said I care about?"

2. MODEL  (learned from your clicks)
   A logistic regression on the embeddings, trained on your feedback:
       thumbs up   -> positive, weight 1.0
       thumbs down -> negative, weight 1.0
       saved       -> positive, weight 1.5
       opened      -> positive, weight 0.3   (weak: a click is not a like)
   Articles you merely scrolled past are NOT used as negatives.
   Its output is a probability 0..1 that you'd like the article.

3. BLEND
       a     = max(0, 1 - n_labels / 50)
       blend = a * prior + (1 - a) * model
   With no labels the prior decides everything; after 50 labelled articles
   the model has fully taken over.

4. FINAL = blend * recency * feed_weight      (clipped to 0..1)
   recency decays mildly from 1.0 (new) towards 0.75 (old) with a 2-day
   half-life; feed_weight is the per-feed weight from feeds.yaml.

5. EXPLORE (applied when a page is built, see interleave_explore)
   About one position in ten is filled from the lower-scored half, favouring
   topics you have given the least feedback on, so the model keeps learning
   about things it would otherwise never show you.

The model is tiny (385 numbers), so we retrain from scratch on every feedback
event and rescore every stored article; no incremental-update cleverness.
The trained weights live in the kv table so both gunicorn workers and the
fetch job always use the same model.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

from . import config, db, embed

# --- tunables ---------------------------------------------------------------

# kind -> (label, sample weight). label 1 = "more like this", 0 = "less".
LABEL_WEIGHTS: dict[str, tuple[int, float]] = {
    "up": (1, 1.0),
    "down": (0, 1.0),
    "save": (1, 1.5),
    "open": (1, 0.3),
}
# If one article has several events, the first kind in this list wins.
LABEL_PRECEDENCE = ("save", "up", "down", "open")

LABELS_FOR_FULL_MODEL = 50  # the "50" in a = max(0, 1 - n/50)

# Cosine similarity of an article to a one-sentence interest description is
# roughly 0.05 (unrelated) .. 0.55 (spot on) for MiniLM. Stretch that to 0..1.
PRIOR_COS_LOW = 0.05
PRIOR_COS_HIGH = 0.55

RECENCY_FLOOR = 0.75  # the oldest article keeps 75% of its score
RECENCY_HALF_LIFE_DAYS = 2.0

# Topic assignment: leave the feed's default topic unless another topic is
# both a decent match and clearly better than the default.
TOPIC_MIN_SIMILARITY = 0.30
TOPIC_SWITCH_MARGIN = 0.12

EXPLORE_EVERY = 10  # one explore slot per this many positions
EXPLORE_FIRST_SLOT = 5  # 0-based position of the first explore slot
EXPLORE_MIN_ARTICLES = 20  # don't explore in a nearly empty list

REGULARISATION_C = 10.0  # LogisticRegression C; higher = trust the labels more


# --- labels -----------------------------------------------------------------


def label_for(kinds: set[str]) -> tuple[int, float] | None:
    """Collapse all feedback on one article into a single (label, weight).

    One article = one training row, so a saved-and-liked article is not
    counted twice. The strongest explicit signal wins (see LABEL_PRECEDENCE).
    """
    for kind in LABEL_PRECEDENCE:
        if kind in kinds:
            return LABEL_WEIGHTS[kind]
    return None


@dataclass
class TrainingSet:
    ids: list[int]
    titles: list[str]
    kinds: list[str]  # the winning feedback kind per row
    X: np.ndarray  # (n, 384) embeddings
    y: np.ndarray  # (n,) 0/1
    w: np.ndarray  # (n,) sample weights

    @property
    def n_labels(self) -> int:
        return len(self.ids)


def load_training_set(conn: sqlite3.Connection) -> TrainingSet:
    """Every article with feedback, as one labelled, weighted row each."""
    rows = conn.execute(
        "SELECT a.id, a.title, a.embedding, GROUP_CONCAT(f.kind) AS kinds "
        "FROM feedback f JOIN articles a ON a.id = f.article_id "
        "WHERE a.embedding IS NOT NULL GROUP BY a.id ORDER BY a.id"
    ).fetchall()
    ids, titles, kinds, vectors, labels, weights = [], [], [], [], [], []
    for row in rows:
        row_kinds = set(row["kinds"].split(","))
        label = label_for(row_kinds)
        if label is None:
            continue
        ids.append(row["id"])
        titles.append(row["title"])
        kinds.append(next(k for k in LABEL_PRECEDENCE if k in row_kinds))
        vectors.append(embed.from_blob(row["embedding"]))
        labels.append(label[0])
        weights.append(label[1])
    X = np.vstack(vectors) if vectors else np.zeros((0, embed.DIM), dtype=np.float32)
    return TrainingSet(ids, titles, kinds, X, np.array(labels, dtype=int), np.array(weights, dtype=float))


# --- the model --------------------------------------------------------------


@dataclass
class Model:
    coef: np.ndarray  # (384,)
    intercept: float
    n_labels: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Probability of "would like" for each row of X."""
        logits = X @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))

    def to_json(self) -> str:
        return json.dumps(
            {"coef": self.coef.tolist(), "intercept": self.intercept, "n_labels": self.n_labels}
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Model":
        data = json.loads(raw)
        return cls(np.array(data["coef"], dtype=np.float32), float(data["intercept"]), int(data["n_labels"]))


def fit_model(train: TrainingSet, background: np.ndarray | None) -> Model | None:
    """Fit the logistic regression. Returns None when there is nothing to learn from.

    Logistic regression needs both classes. Early on you will often have only
    likes (or only dislikes). In that case we add ONE synthetic row for the
    missing class: the average embedding of the whole corpus ("a typical
    article"). The model then learns the direction from "typical" towards
    "what you liked" -- without pretending any specific article you skipped
    was a dislike. The synthetic row is dropped as soon as real examples of
    both classes exist.
    """
    if train.n_labels == 0:
        return None
    X, y, w = train.X, train.y, train.w
    if len(set(y.tolist())) < 2:
        if background is None:
            return None
        missing_class = 1 - int(y[0])
        X = np.vstack([X, background[None, :]])
        y = np.append(y, missing_class)
        w = np.append(w, 1.0)

    from sklearn.linear_model import LogisticRegression

    # class_weight="balanced" stops 40 likes + 2 dislikes from producing a
    # model that just says "yes" to everything.
    clf = LogisticRegression(C=REGULARISATION_C, class_weight="balanced", max_iter=1000)
    clf.fit(X, y, sample_weight=w)
    return Model(clf.coef_[0].astype(np.float32), float(clf.intercept_[0]), train.n_labels)


def save_model(conn: sqlite3.Connection, model: Model | None) -> None:
    if model is None:
        conn.execute("DELETE FROM kv WHERE key = 'model'")
    else:
        db.kv_set(conn, "model", model.to_json())


def load_model(conn: sqlite3.Connection) -> Model | None:
    raw = db.kv_get(conn, "model")
    return Model.from_json(raw) if raw else None


# --- score maths (pure functions, unit tested) ------------------------------


def blend_alpha(n_labels: int) -> float:
    """Weight of the prior: 1.0 with no labels, 0.0 from 50 labels on."""
    return max(0.0, 1.0 - n_labels / LABELS_FOR_FULL_MODEL)


def blend(prior: float, model_score: float | None, n_labels: int) -> float:
    """final = a*prior + (1-a)*model. Without a model the prior stands alone."""
    if model_score is None:
        return prior
    a = blend_alpha(n_labels)
    return a * prior + (1.0 - a) * model_score


def prior_from_cosine(cosine: float) -> float:
    """Rescale a raw cosine similarity to a 0..1 prior."""
    scaled = (cosine - PRIOR_COS_LOW) / (PRIOR_COS_HIGH - PRIOR_COS_LOW)
    return float(min(1.0, max(0.0, scaled)))


def recency_factor(age_days: float) -> float:
    """1.0 for a brand-new article, decaying towards RECENCY_FLOOR."""
    age_days = max(0.0, age_days)
    return RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def final_score(blended: float, age_days: float, feed_weight: float) -> float:
    return float(min(1.0, max(0.0, blended * recency_factor(age_days) * feed_weight)))


# --- interests: the prior and topic assignment ------------------------------


@dataclass
class Interests:
    labels: list[str]  # e.g. "F1 engines"
    vectors: np.ndarray  # (n_interests, 384)
    topic_names: list[str]
    topic_of_vector: list[str]  # topic each row of topic_vectors belongs to
    topic_vectors: np.ndarray  # "about" line + interest lines of every topic


def load_interests(conn: sqlite3.Connection, embedder: embed.Embedder) -> tuple[Interests, bool]:
    """Embed config/interests.yaml (cached in kv until the file changes).

    Returns (interests, changed); changed=True means priors must be recomputed.
    """
    topics = config.load_topics()
    payload = json.dumps([[t.name, t.about, t.interests] for t in topics], sort_keys=True)
    digest = hashlib.sha256(f"{embedder.name}:{payload}".encode()).hexdigest()

    labels = [label for t in topics for label in t.interests]
    topic_of_vector = [t.name for t in topics for _ in range(1 + len(t.interests))]
    cached_digest = db.kv_get(conn, "interests:digest")
    cached = db.kv_get(conn, "interests:vectors")
    if cached and cached_digest and _as_str(cached_digest) == digest:
        matrix = embed.from_blob(cached).reshape(-1, embed.DIM)
        changed = False
    else:
        interest_texts = [f"{label}: {text}" for t in topics for label, text in t.interests.items()]
        topic_texts = [
            text for t in topics for text in [t.about, *[f"{k}: {v}" for k, v in t.interests.items()]]
        ]
        matrix = embedder.embed(interest_texts + topic_texts)
        db.kv_set(conn, "interests:vectors", embed.to_blob(matrix))
        db.kv_set(conn, "interests:digest", digest)
        conn.commit()
        changed = True
    n = len(labels)
    return Interests(labels, matrix[:n], [t.name for t in topics], topic_of_vector, matrix[n:]), changed


def _as_str(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def assign_topic(vector: np.ndarray, default_topic: str, interests: Interests) -> str:
    """The feed's default topic, unless the embedding clearly says otherwise."""
    if not len(interests.topic_vectors):
        return default_topic
    sims = interests.topic_vectors @ vector
    best_per_topic: dict[str, float] = {}
    for topic, sim in zip(interests.topic_of_vector, sims):
        best_per_topic[topic] = max(best_per_topic.get(topic, -1.0), float(sim))
    best_topic = max(best_per_topic, key=lambda t: best_per_topic[t])
    default_sim = best_per_topic.get(default_topic, -1.0)
    clearly_better = best_per_topic[best_topic] - default_sim >= TOPIC_SWITCH_MARGIN
    if best_topic != default_topic and best_per_topic[best_topic] >= TOPIC_MIN_SIMILARITY and clearly_better:
        return best_topic
    return default_topic


def update_priors(conn: sqlite3.Connection, interests: Interests, recompute_all: bool = False) -> int:
    """Compute prior, matched interest and topic for articles that lack them."""
    where = "a.embedding IS NOT NULL" + ("" if recompute_all else " AND a.prior IS NULL")
    rows = conn.execute(
        f"SELECT a.id, a.embedding, f.default_topic FROM articles a JOIN feeds f ON f.id = a.feed_id WHERE {where}"
    ).fetchall()
    updates = []
    for row in rows:
        vector = embed.from_blob(row["embedding"])
        if len(interests.labels):
            sims = interests.vectors @ vector
            best = int(np.argmax(sims))
            prior, label = prior_from_cosine(float(sims[best])), interests.labels[best]
        else:
            prior, label = 0.5, None
        topic = assign_topic(vector, row["default_topic"], interests)
        updates.append((prior, label, topic, row["id"]))
    conn.executemany("UPDATE articles SET prior = ?, prior_label = ?, topic = ? WHERE id = ?", updates)
    conn.commit()
    return len(updates)


# --- "why this?" ------------------------------------------------------------

_KIND_PHRASE = {"save": "you saved", "up": "you liked", "open": "you opened", "down": "you disliked"}
NEIGHBOUR_MIN_SIMILARITY = 0.45


def _short(title: str, limit: int = 42) -> str:
    return title if len(title) <= limit else title[:limit].rsplit(" ", 1)[0] + "…"


def explain(
    alpha: float,
    prior: float,
    prior_label: str | None,
    model_score: float | None,
    neighbour: tuple[str, str, float] | None,
) -> str:
    """One honest line about what drove this article's score.

    We report whichever term of the blend contributed more:
      * prior term  (a * prior)       -> name the interest it matched;
      * model term  ((1-a) * model)   -> name the most similar article you gave
        feedback on. The logistic regression is a single linear direction, so
        "your nearest labelled article" is an approximation of its reasoning,
        but it is built from exactly the examples the model was trained on.
    neighbour is (title, feedback kind, cosine similarity) or None.
    """
    prior_part = alpha * prior
    model_part = 0.0 if model_score is None else (1.0 - alpha) * model_score
    if model_score is None or prior_part >= model_part:
        if prior_label is None:
            return "new for you"
        if prior >= 0.35:
            return f'matches your interest "{prior_label}"'
        return f'loosely related to "{prior_label}"'
    if neighbour and neighbour[2] >= NEIGHBOUR_MIN_SIMILARITY:
        title, kind, _ = neighbour
        return f'similar to "{_short(title)}" {_KIND_PHRASE[kind]}'
    if model_score >= 0.5:
        return "fits the pattern of what you like"
    return "not much like what you've liked so far"


# --- scoring everything -----------------------------------------------------


def rescore_all(conn: sqlite3.Connection, now: float | None = None) -> int:
    """Recompute score, model_score and reason for every embedded article."""
    now = time.time() if now is None else now
    model = load_model(conn)
    train = load_training_set(conn)
    rows = conn.execute(
        "SELECT a.id, a.embedding, a.prior, a.prior_label, a.published_at, f.weight "
        "FROM articles a JOIN feeds f ON f.id = a.feed_id "
        "WHERE a.embedding IS NOT NULL AND a.prior IS NOT NULL"
    ).fetchall()
    if not rows:
        return 0
    X = np.vstack([embed.from_blob(r["embedding"]) for r in rows])
    model_scores = model.predict(X) if model is not None else None
    n_labels = model.n_labels if model is not None else 0
    alpha = blend_alpha(n_labels)

    # Nearest labelled article for each row (only used for the reason line).
    nearest_idx = nearest_sim = None
    if train.n_labels and model is not None:
        sims = X @ train.X.T  # (n_articles, n_labelled) cosine similarities
        id_to_col = {article_id: col for col, article_id in enumerate(train.ids)}
        for i, row in enumerate(rows):  # an article is not its own neighbour
            col = id_to_col.get(row["id"])
            if col is not None:
                sims[i, col] = -1.0
        nearest_idx = sims.argmax(axis=1)
        nearest_sim = sims.max(axis=1)

    updates = []
    for i, row in enumerate(rows):
        m = None if model_scores is None else float(model_scores[i])
        blended = blend(row["prior"], m, n_labels)
        age_days = (now - row["published_at"]) / 86400.0
        score = final_score(blended, age_days, row["weight"])
        neighbour = None
        if nearest_idx is not None:
            j = int(nearest_idx[i])
            neighbour = (train.titles[j], train.kinds[j], float(nearest_sim[i]))
        reason = explain(alpha, row["prior"], row["prior_label"], m, neighbour)
        updates.append((score, m, reason, row["id"]))
    conn.executemany("UPDATE articles SET score = ?, model_score = ?, reason = ? WHERE id = ?", updates)
    conn.commit()
    return len(updates)


def corpus_mean(conn: sqlite3.Connection, sample: int = 2000) -> np.ndarray | None:
    """Average embedding of (a sample of) all articles: the "typical article"."""
    rows = conn.execute(
        "SELECT embedding FROM articles WHERE embedding IS NOT NULL ORDER BY id DESC LIMIT ?", (sample,)
    ).fetchall()
    if not rows:
        return None
    mean = np.vstack([embed.from_blob(r[0]) for r in rows]).mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm else None


def retrain(conn: sqlite3.Connection) -> Model | None:
    """Train from all current feedback, store the model, rescore everything."""
    train = load_training_set(conn)
    model = fit_model(train, corpus_mean(conn))
    save_model(conn, model)
    conn.commit()
    rescore_all(conn)
    return model


# --- building a page of results ---------------------------------------------


def interleave_explore(
    ranked: list[tuple[int, str]],
    label_counts: dict[str, int],
    seed: str,
) -> list[tuple[int, bool]]:
    """Insert explore picks into a best-first list.

    ranked        [(article_id, topic)] sorted best first.
    label_counts  topic -> number of labelled articles ("how well do I know it").
    seed          makes the picks stable across "load more" requests.

    Returns [(article_id, is_explore)]. Positions 5, 15, 25, ... (0-based) are
    explore slots. Each takes an article from the lower-scored half of the
    list, cycling through topics from least-labelled to most-labelled so the
    topics the model knows least about get shown first. Everything else keeps
    its order; no article appears twice.
    """
    n = len(ranked)
    if n < EXPLORE_MIN_ARTICLES:
        return [(article_id, False) for article_id, _ in ranked]

    rng = random.Random(seed)
    pool_by_topic: dict[str, list[int]] = {}
    for article_id, topic in ranked[n // 2 :]:
        pool_by_topic.setdefault(topic, []).append(article_id)
    topics = sorted(pool_by_topic, key=lambda t: (label_counts.get(t, 0), t))

    n_slots = len(range(EXPLORE_FIRST_SLOT, n, EXPLORE_EVERY))
    picks: list[int] = []
    for k in range(n_slots):
        for step in range(len(topics)):
            candidates = pool_by_topic[topics[(k + step) % len(topics)]]
            if candidates:
                # choose among the better part of that topic's pool
                choice = rng.choice(candidates[: max(1, min(20, len(candidates)))])
                candidates.remove(choice)
                picks.append(choice)
                break
    picked = set(picks)
    main = iter([article_id for article_id, _ in ranked if article_id not in picked])
    pick_iter = iter(picks)

    result: list[tuple[int, bool]] = []
    for position in range(n):
        is_slot = position >= EXPLORE_FIRST_SLOT and (position - EXPLORE_FIRST_SLOT) % EXPLORE_EVERY == 0
        article_id = next(pick_iter, None) if is_slot else None
        if article_id is not None:
            result.append((article_id, True))
            continue
        article_id = next(main, None)
        if article_id is None:  # main list exhausted; flush remaining picks
            article_id = next(pick_iter, None)
            if article_id is None:
                break
            result.append((article_id, True))
        else:
            result.append((article_id, False))
    return result


def topic_label_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT a.topic, COUNT(DISTINCT a.id) FROM feedback f JOIN articles a ON a.id = f.article_id GROUP BY a.topic"
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def build_feed(
    conn: sqlite3.Connection,
    tab: str,
    offset: int,
    limit: int,
    day: str | None = None,
    exclude: set[int] | None = None,
) -> tuple[list[tuple[int, bool]], bool]:
    """One page of [(article_id, is_explore)] for a tab, plus has_more.

    tab is "for-you", "saved" or a topic name. Disliked articles and muted or
    removed feeds are left out (the Saved tab shows everything you saved).

    exclude holds article ids the browser already shows. "Load more" passes
    them instead of a numeric offset, because every feedback click retrains
    and reorders the list: with plain offsets some articles would be shown
    twice and others never.
    """
    if tab == "saved":
        rows = conn.execute(
            "SELECT a.id FROM feedback f JOIN articles a ON a.id = f.article_id "
            "WHERE f.kind = 'save' ORDER BY f.created_at DESC, a.id DESC"
        ).fetchall()
        ordered = [(row[0], False) for row in rows]
    else:
        sql = (
            "SELECT a.id, a.topic FROM articles a JOIN feeds f ON f.id = a.feed_id "
            "WHERE a.score IS NOT NULL AND f.muted = 0 AND f.removed = 0 "
            "AND NOT EXISTS (SELECT 1 FROM feedback d WHERE d.article_id = a.id AND d.kind = 'down') "
        )
        params: tuple = ()
        if tab != "for-you":
            sql += "AND a.topic = ? "
            params = (tab,)
        sql += "ORDER BY a.score DESC, a.id DESC"
        ranked = [(row[0], row[1]) for row in conn.execute(sql, params).fetchall()]
        day = day or time.strftime("%Y-%m-%d")
        ordered = interleave_explore(ranked, topic_label_counts(conn), seed=f"{day}:{tab}")
    if exclude:
        ordered = [item for item in ordered if item[0] not in exclude]
    page = ordered[offset : offset + limit]
    return page, offset + limit < len(ordered)

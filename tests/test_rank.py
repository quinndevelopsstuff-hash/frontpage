import numpy as np
import pytest

from core import embed, rank


# --- blend maths ------------------------------------------------------------


def test_alpha_schedule():
    assert rank.blend_alpha(0) == 1.0
    assert rank.blend_alpha(10) == pytest.approx(0.8)
    assert rank.blend_alpha(25) == pytest.approx(0.5)
    assert rank.blend_alpha(50) == 0.0
    assert rank.blend_alpha(500) == 0.0


def test_blend_is_a_times_prior_plus_rest_times_model():
    assert rank.blend(0.8, 0.2, 0) == pytest.approx(0.8)  # all prior
    assert rank.blend(0.8, 0.2, 25) == pytest.approx(0.5 * 0.8 + 0.5 * 0.2)
    assert rank.blend(0.8, 0.2, 10) == pytest.approx(0.8 * 0.8 + 0.2 * 0.2)
    assert rank.blend(0.8, 0.2, 50) == pytest.approx(0.2)  # all model
    assert rank.blend(0.8, 0.2, 80) == pytest.approx(0.2)


def test_blend_without_a_model_is_the_prior():
    assert rank.blend(0.63, None, 30) == 0.63


def test_prior_rescaling_is_clamped():
    assert rank.prior_from_cosine(-0.2) == 0.0
    assert rank.prior_from_cosine(rank.PRIOR_COS_LOW) == 0.0
    assert rank.prior_from_cosine(0.30) == pytest.approx(0.5)
    assert rank.prior_from_cosine(0.9) == 1.0


def test_recency_is_mild_and_monotonic():
    assert rank.recency_factor(0) == 1.0
    assert rank.recency_factor(-3) == 1.0
    assert rank.recency_factor(2) == pytest.approx(0.875)  # one half-life
    assert rank.recency_factor(1) > rank.recency_factor(5) > rank.recency_factor(30) > rank.RECENCY_FLOOR


def test_final_score_applies_recency_and_feed_weight():
    assert rank.final_score(0.8, 0, 1.0) == pytest.approx(0.8)
    assert rank.final_score(0.8, 2, 0.5) == pytest.approx(0.8 * 0.875 * 0.5)
    assert rank.final_score(0.9, 0, 2.0) == 1.0  # clipped


# --- label weighting --------------------------------------------------------


def test_label_weights_match_the_spec():
    assert rank.label_for({"up"}) == (1, 1.0)
    assert rank.label_for({"down"}) == (0, 1.0)
    assert rank.label_for({"save"}) == (1, 1.5)
    assert rank.label_for({"open"}) == (1, 0.3)
    assert rank.label_for(set()) is None


def test_strongest_signal_wins_and_articles_count_once():
    assert rank.label_for({"open", "up"}) == (1, 1.0)
    assert rank.label_for({"open", "up", "save"}) == (1, 1.5)
    assert rank.label_for({"open", "down"}) == (0, 1.0)  # opened, then disliked


def _article_id(conn, fragment):
    return conn.execute("SELECT id FROM articles WHERE title LIKE ?", (f"%{fragment}%",)).fetchone()[0]


def _feedback(conn, article_id, kind):
    conn.execute("INSERT INTO feedback (article_id, kind, created_at) VALUES (?, ?, 0)", (article_id, kind))
    conn.commit()


def test_training_set_uses_weights_and_ignores_unlabelled(seeded):
    liked, saved, opened, disliked = (
        _article_id(seeded, "engine race"),
        _article_id(seeded, "tyre wear"),
        _article_id(seeded, "floor upgrade"),
        _article_id(seeded, "thermal camera"),
    )
    _feedback(seeded, liked, "up")
    _feedback(seeded, saved, "save")
    _feedback(seeded, saved, "open")
    _feedback(seeded, opened, "open")
    _feedback(seeded, disliked, "down")
    train = rank.load_training_set(seeded)
    got = {i: (int(y), float(w)) for i, y, w in zip(train.ids, train.y, train.w)}
    assert got == {liked: (1, 1.0), saved: (1, 1.5), opened: (1, 0.3), disliked: (0, 1.0)}
    assert train.n_labels == 4  # shown-but-ignored articles are not in the set


# --- the model --------------------------------------------------------------


def test_no_feedback_means_no_model(seeded):
    assert rank.retrain(seeded) is None
    row = seeded.execute("SELECT score, prior, model_score FROM articles LIMIT 1").fetchone()
    assert row["model_score"] is None and row["score"] is not None


def test_model_learns_from_feedback(seeded):
    for fragment in ("engine race", "tyre wear", "floor upgrade", "flexible wings"):
        _feedback(seeded, _article_id(seeded, fragment), "up")
    for fragment in ("thermal camera", "Printing gears", "motor controller"):
        _feedback(seeded, _article_id(seeded, fragment), "down")
    model = rank.retrain(seeded)
    assert model is not None and model.n_labels == 7
    scores = dict(seeded.execute("SELECT title, model_score FROM articles").fetchall())
    unseen_f1 = next(v for k, v in scores.items() if "Qualifying pace" in k)
    unseen_eng = next(v for k, v in scores.items() if "battery pack" in k)
    assert unseen_f1 > unseen_eng
    # the stored model is what a new process would load
    assert rank.load_model(seeded).n_labels == 7


def test_only_likes_still_trains(seeded):
    """With one class only, the 'typical article' row stands in for the other."""
    for fragment in ("engine race", "tyre wear", "floor upgrade"):
        _feedback(seeded, _article_id(seeded, fragment), "up")
    model = rank.retrain(seeded)
    assert model is not None
    scores = dict(seeded.execute("SELECT title, model_score FROM articles").fetchall())
    liked = next(v for k, v in scores.items() if "tyre wear" in k)
    unrelated = next(v for k, v in scores.items() if "thermal camera" in k)
    assert liked > unrelated


def test_rescore_matches_the_formula(seeded):
    for fragment in ("engine race", "tyre wear"):
        _feedback(seeded, _article_id(seeded, fragment), "up")
    _feedback(seeded, _article_id(seeded, "thermal camera"), "down")
    model = rank.retrain(seeded)
    now = seeded.execute("SELECT MAX(fetched_at) FROM articles").fetchone()[0]
    rank.rescore_all(seeded, now=now)
    for row in seeded.execute(
        "SELECT a.score, a.prior, a.model_score, a.published_at, f.weight FROM articles a JOIN feeds f ON f.id = a.feed_id"
    ):
        a = rank.blend_alpha(model.n_labels)
        expected = (a * row["prior"] + (1 - a) * row["model_score"]) * rank.recency_factor(
            (now - row["published_at"]) / 86400
        ) * row["weight"]
        assert row["score"] == pytest.approx(min(1.0, expected), abs=1e-6)


# --- explanations -----------------------------------------------------------


def test_explain_reports_the_dominant_term():
    # cold start: prior dominates
    assert rank.explain(1.0, 0.8, "F1 engines", None, None) == 'matches your interest "F1 engines"'
    assert rank.explain(0.9, 0.2, "systems", 0.5, None) == 'loosely related to "systems"'
    # model dominates and a close labelled neighbour exists
    reason = rank.explain(0.1, 0.5, "F1 engines", 0.9, ("A tyre story", "save", 0.7))
    assert reason == 'similar to "A tyre story" you saved'
    # model dominates but nothing labelled is close: no invented neighbour
    assert rank.explain(0.0, 0.5, "F1 engines", 0.9, ("Far away", "up", 0.1)) == "fits the pattern of what you like"
    assert rank.explain(0.0, 0.0, None, 0.2, None) == "not much like what you've liked so far"


# --- topics -----------------------------------------------------------------


def test_topic_switches_only_when_clearly_better():
    unit = np.eye(embed.DIM, dtype=np.float32)
    interests = rank.Interests([], np.zeros((0, embed.DIM), dtype=np.float32), ["F1", "AI"], ["F1", "AI"], unit[:2])
    assert rank.assign_topic(unit[1], "F1", interests) == "AI"  # sim 1.0 vs 0.0
    near_tie = (unit[0] * 0.70 + unit[1] * 0.71).astype(np.float32)
    assert rank.assign_topic(near_tie, "F1", interests) == "F1"  # not clearly better
    weak = (unit[1] * 0.2 + unit[5] * 0.98).astype(np.float32)
    assert rank.assign_topic(weak, "F1", interests) == "F1"  # below minimum similarity


# --- explore ----------------------------------------------------------------


def _ranked(n):
    topics = ["F1", "AI", "CS", "Engineering"]
    return [(i, topics[i % 4]) for i in range(n)]


def test_explore_slots_are_about_one_in_ten():
    result = rank.interleave_explore(_ranked(100), {"F1": 9, "AI": 5, "CS": 0, "Engineering": 2}, "seed")
    assert sorted(i for i, _ in result) == list(range(100))  # nothing lost or repeated
    explore_positions = [pos for pos, (_, flag) in enumerate(result) if flag]
    assert explore_positions == list(range(5, 100, 10))
    assert not result[0][1]  # the hero is never an explore pick


def test_explore_picks_come_from_the_lower_half_least_labelled_first():
    ranked = _ranked(100)
    result = rank.interleave_explore(ranked, {"F1": 9, "AI": 5, "CS": 0, "Engineering": 2}, "seed")
    picks = [article_id for article_id, flag in result if flag]
    assert all(article_id >= 50 for article_id in picks)
    topic_of = dict(ranked)
    assert [topic_of[p] for p in picks[:4]] == ["CS", "Engineering", "AI", "F1"]


def test_explore_is_stable_for_a_seed_and_skipped_for_tiny_lists():
    counts = {"F1": 1}
    assert rank.interleave_explore(_ranked(60), counts, "a") == rank.interleave_explore(_ranked(60), counts, "a")
    assert all(not flag for _, flag in rank.interleave_explore(_ranked(10), counts, "a"))


def test_build_feed_hides_disliked_and_muted(seeded):
    disliked = _article_id(seeded, "thermal camera")
    _feedback(seeded, disliked, "down")
    muted_feed = seeded.execute("SELECT feed_id FROM articles WHERE title LIKE '%engine race%'").fetchone()[0]
    seeded.execute("UPDATE feeds SET muted = 1 WHERE id = ?", (muted_feed,))
    seeded.commit()
    page, _ = rank.build_feed(seeded, "for-you", 0, 100)
    ids = {article_id for article_id, _ in page}
    assert disliked not in ids
    muted_ids = {r[0] for r in seeded.execute("SELECT id FROM articles WHERE feed_id = ?", (muted_feed,))}
    assert muted_ids and not (muted_ids & ids)


def test_build_feed_topic_and_saved_tabs(seeded):
    page, has_more = rank.build_feed(seeded, "F1", 0, 3)
    assert len(page) == 3 and has_more
    topics = {r[0] for r in seeded.execute(f"SELECT topic FROM articles WHERE id IN ({','.join(str(i) for i, _ in page)})")}
    assert topics == {"F1"}
    saved = _article_id(seeded, "tyre wear")
    _feedback(seeded, saved, "save")
    assert rank.build_feed(seeded, "saved", 0, 10) == ([(saved, False)], False)

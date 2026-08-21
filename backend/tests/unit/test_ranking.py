"""Ranking, and the line between ordering and deciding.

Hybrid search fuses a vector arm and a full-text arm with reciprocal rank
fusion. RRF is excellent at ordering and useless at judging: a fused score of
0.0163 means "this came first", not "this is right", and it cannot tell a
perfect match from the best of a bad lot. Two lists of garbage fuse into one
ordered list of garbage with a confident-looking number on top.

So every decision — is this worth showing, are the top two too close to call, is
this good enough to write against — is made on `cn` (cosine normalised per
corpus, z-scored, clamped to 0..1) and on the `evidence` flag, never on the
fused score. The two tests in the middle of this file are the ones that prove
it: a pair of hits with a hair's-breadth RRF gap and a wide `cn` gap is not
ambiguous, and the reverse is.

Then scoring applies time. Mail decays — `score * exp(-age_days / 30)` — because
an old email is usually the wrong one. Events do the opposite: something coming
up matters more than something that already happened.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from tests.conftest import FLOOR_READ, FLOOR_WRITE, FROZEN_NOW, MARGIN, call, get, load_any

_RANKING_MODULES = ["app.search.hybrid", "app.search.probe", "app.search.scoring"]

rrf_fuse = load_any(_RANKING_MODULES, ["rrf_fuse", "fuse_rrf", "rrf", "fuse"])
cn_scores = load_any(_RANKING_MODULES, ["cn_scores", "normalise_cn", "normalize_cn", "to_cn"])
is_ambiguous = load_any(_RANKING_MODULES, ["is_ambiguous", "ambiguous", "check_ambiguity"])
qualifies = load_any(
    _RANKING_MODULES, ["qualifies", "meets_floor", "passes_floor", "above_floor", "admits"]
)
decay = load_any(_RANKING_MODULES, ["temporal_decay", "decay", "decay_score", "apply_decay"])
forward_boost = load_any(
    _RANKING_MODULES, ["forward_boost", "event_boost", "future_boost", "apply_forward_boost"]
)


# ---------------------------------------------------------------------------
# Small readers, so one test works whatever shape the function hands back
# ---------------------------------------------------------------------------


def ordered_ids(result) -> list[str]:
    """The ids in the order the fusion put them."""
    if isinstance(result, dict):
        return [k for k, _ in sorted(result.items(), key=lambda kv: -kv[1])]
    out = []
    for item in result:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (tuple, list)):
            out.append(item[0])
        else:
            out.append(get(item, "id"))
    return out


def score_for(result, wanted: str) -> float:
    """The fused score attached to one id."""
    if isinstance(result, dict):
        return float(result[wanted])
    for item in result:
        if isinstance(item, (tuple, list)) and item[0] == wanted:
            return float(item[1])
        if not isinstance(item, str) and get(item, "id", None) == wanted:
            for name in ("score", "rrf", "fused", "rrf_score", "fused_score"):
                value = get(item, name, None)
                if value is not None:
                    return float(value)
    raise AssertionError(f"no fused score for {wanted!r} in {result!r}")


def fuse(*rankings, **kw):
    """Fuse ranked id lists, passing them however the function wants them."""
    try:
        return call(rrf_fuse, list(rankings), **kw)
    except (TypeError, AttributeError, KeyError, IndexError):
        return call(rrf_fuse, *rankings, **kw)


def hit(hit_id: str, cn: float, *, score: float = 0.0, evidence=None, **extra) -> dict:
    """One candidate as the probe hands it to the planner."""
    return {
        "id": hit_id,
        "cn": cn,
        "score": score,
        "rrf": score,
        "evidence": evidence,
        **extra,
    }


def ambiguous(hits, expect="one", margin=MARGIN) -> bool:
    return bool(call(is_ambiguous, hits, expect=expect, margin=margin))


def qualifying(candidate, floor=FLOOR_READ) -> bool:
    return bool(call(qualifies, candidate, floor=floor))


def decayed(score: float, age_days: float) -> float:
    """Mail decay, however the function likes its age."""
    try:
        return float(call(decay, score, age_days))
    except (TypeError, ValueError):
        received = FROZEN_NOW - timedelta(days=age_days)
        return float(call(decay, score, received, now=FROZEN_NOW))


def boosted(score: float, days_ahead: float) -> float:
    """Event boost, however the function likes its horizon."""
    try:
        return float(call(forward_boost, score, days_ahead))
    except (TypeError, ValueError):
        starts_at = FROZEN_NOW + timedelta(days=days_ahead)
        return float(call(forward_boost, score, starts_at, now=FROZEN_NOW))


# ---------------------------------------------------------------------------
# RRF: ordering
# ---------------------------------------------------------------------------


def test_agreement_across_both_arms_beats_a_win_in_one():
    # `a` is second on both lists. `x` is first on the vector arm and absent
    # from the text arm; `y` is the mirror image. Two second places beat one
    # first place, which is the property RRF exists for.
    vector = ["x", "a", "b"]
    lexical = ["y", "a", "b"]
    order = ordered_ids(fuse(vector, lexical))

    assert order[0] == "a", order
    assert order.index("a") < order.index("x"), order
    assert order.index("a") < order.index("y"), order
    # `b` is only third on both lists, and even that beats one first place.
    assert order.index("b") < order.index("x"), order


def test_the_default_k_is_sixty():
    # Rank 1 in one list only: 1 / (60 + 1).
    result = fuse(["a"], [])
    assert score_for(result, "a") == pytest.approx(1.0 / 61.0, rel=1e-6)


def test_a_document_missing_from_an_arm_contributes_nothing_from_it():
    # Not 1/(k+0), and not a penalty either. It simply is not in that list.
    both = score_for(fuse(["a"], ["a"]), "a")
    one = score_for(fuse(["a"], ["z"]), "a")
    assert both == pytest.approx(2.0 / 61.0, rel=1e-6)
    assert one == pytest.approx(1.0 / 61.0, rel=1e-6)


def test_deeper_ranks_are_worth_steadily_less():
    result = fuse(["a", "b", "c", "d"], [])
    scores = [score_for(result, i) for i in ("a", "b", "c", "d")]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1 / 61, rel=1e-6)
    assert scores[3] == pytest.approx(1 / 64, rel=1e-6)
    # The gap between first and fourth is under a thousandth. That is the whole
    # problem with treating this number as a judgement.
    assert scores[0] - scores[3] < 0.001


def test_fusion_is_deterministic():
    first = ordered_ids(fuse(["a", "b", "c"], ["c", "b", "a"]))
    second = ordered_ids(fuse(["a", "b", "c"], ["c", "b", "a"]))
    assert first == second


def test_a_perfect_match_still_scores_far_below_the_read_floor():
    # The single most important fact about the fused score. A document that came
    # first in both arms scores about 0.033, which is under FLOOR_READ 0.55 and
    # under FLOOR_WRITE 0.80. Compare a fused score to a threshold and every
    # candidate in the system fails, forever.
    best = score_for(fuse(["a", "b"], ["a", "b"]), "a")
    assert best == pytest.approx(2.0 / 61.0, rel=1e-6)
    assert best < FLOOR_READ
    assert best < FLOOR_WRITE


# ---------------------------------------------------------------------------
# cn normalisation
# ---------------------------------------------------------------------------


def test_cn_keeps_the_order_of_the_raw_cosines():
    raw = [0.91, 0.74, 0.62, 0.38, 0.11]
    out = list(call(cn_scores, raw))
    assert len(out) == len(raw)
    assert out == sorted(out, reverse=True), out


def test_cn_stays_inside_zero_and_one():
    for raw in ([0.91, 0.74, 0.62], [0.02, 0.03], [0.99, 0.98, 0.97, 0.01]):
        for value in call(cn_scores, raw):
            assert 0.0 <= value <= 1.0, (raw, value)


def test_cn_of_identical_cosines_is_identical():
    # Z-scoring a flat list must not divide by zero and must not invent a
    # ranking out of noise.
    out = list(call(cn_scores, [0.7, 0.7, 0.7]))
    assert len(set(out)) == 1
    assert 0.0 <= out[0] <= 1.0


def test_cn_of_a_single_candidate_is_defined():
    out = list(call(cn_scores, [0.83]))
    assert len(out) == 1
    assert 0.0 <= out[0] <= 1.0


# ---------------------------------------------------------------------------
# The floor is a disjunction
# ---------------------------------------------------------------------------


def test_a_candidate_above_the_read_floor_qualifies():
    assert qualifying(hit("m1", cn=0.60))


def test_a_candidate_below_the_floor_with_nothing_else_does_not():
    assert not qualifying(hit("m1", cn=0.41))


def test_a_candidate_below_the_floor_with_exact_evidence_still_qualifies():
    # docs/SAMPLE_QUERIES.md §12: the Turkish booking email. `cn` is 0.41
    # because an English query does not embed near a Turkish body, and the
    # sender is `bilet@thy.com` with `TK1984` in the subject. Two independent
    # exact signals beat a similarity number every time.
    candidate = hit("m1", cn=0.41, evidence="EXACT(sender-domain, code-pattern)")
    assert qualifying(candidate)


def test_the_write_floor_is_stricter_than_the_read_floor():
    candidate = hit("m1", cn=0.62)
    assert qualifying(candidate, floor=FLOOR_READ)
    assert not qualifying(candidate, floor=FLOOR_WRITE)


# ---------------------------------------------------------------------------
# Ambiguity — decided on cn, not on the fused score
# ---------------------------------------------------------------------------


def test_a_tiny_rrf_gap_with_a_wide_cn_gap_is_not_ambiguous():
    # docs/SAMPLE_QUERIES.md §4. The booking confirmation at cn 0.88 and a
    # marketing email at cn 0.61 — 0.27 apart, nobody would call that a tie.
    # Their fused scores differ by 0.0001, because they are ranks 1 and 2.
    hits = [
        hit("booking", cn=0.88, score=0.03279, evidence="EXACT(alias-token-in-subject)"),
        hit("promo", cn=0.61, score=0.03269),
    ]
    assert hits[0]["score"] - hits[1]["score"] < 0.001  # the trap
    assert hits[0]["cn"] - hits[1]["cn"] > MARGIN
    assert not ambiguous(hits, expect="one")


def test_a_wide_rrf_gap_with_a_tiny_cn_gap_is_ambiguous():
    # docs/SAMPLE_QUERIES.md §7, "Move the meeting with John". 0.72 against
    # 0.68 is a coin toss, and the fused scores are far apart only because one
    # of them had to be printed first.
    hits = [
        hit("okafor", cn=0.72, score=0.0328),
        hit("reyes", cn=0.68, score=0.0100),
    ]
    assert hits[0]["score"] - hits[1]["score"] > 0.02  # the other trap
    assert hits[0]["cn"] - hits[1]["cn"] < MARGIN
    assert ambiguous(hits, expect="one")


def test_exactly_at_the_margin_is_not_ambiguous():
    # The comparison is `gap < MARGIN`, so a gap of exactly 0.15 is decided.
    hits = [hit("a", cn=0.80), hit("b", cn=0.65)]
    assert not ambiguous(hits, expect="one")


def test_expecting_many_never_raises_ambiguity():
    # "Find emails from sarah about the budget" wants the list. Two close
    # candidates are two results, not a question.
    hits = [hit("a", cn=0.72), hit("b", cn=0.71), hit("c", cn=0.70)]
    assert ambiguous(hits, expect="one")
    assert not ambiguous(hits, expect="many")


def test_one_candidate_is_never_ambiguous():
    assert not ambiguous([hit("a", cn=0.72)], expect="one")


def test_no_candidates_is_absence_not_ambiguity():
    # Nothing found is answered with "I could not find it", not with a card
    # offering a choice between no options.
    assert not ambiguous([], expect="one")


def test_exact_evidence_on_the_leader_alone_settles_a_close_pair():
    # Same 0.04 gap as the John case, but this time the top hit matches on the
    # sender address and the runner-up matches on nothing. That is not a tie,
    # and asking would be asking a question we already know the answer to.
    hits = [
        hit("from_sarah", cn=0.72, evidence="EXACT(sender)"),
        hit("mentions_sarah", cn=0.68),
    ]
    assert not ambiguous(hits, expect="one")


def test_exact_evidence_on_both_leaves_it_ambiguous():
    # Two real Turkish Airlines bookings. Both match exactly; that is precisely
    # why the user has to pick. docs/SAMPLE_QUERIES.md §10.
    hits = [
        hit("sep_flight", cn=0.84, evidence="EXACT(sender-domain)"),
        hit("oct_flight", cn=0.82, evidence="EXACT(sender-domain)"),
    ]
    assert ambiguous(hits, expect="one")


# ---------------------------------------------------------------------------
# Time: mail decays, events lean forward
# ---------------------------------------------------------------------------


def test_mail_decay_is_the_stated_curve():
    assert decayed(1.0, 0) == pytest.approx(1.0, rel=1e-6)
    assert decayed(1.0, 30) == pytest.approx(math.exp(-1.0), rel=1e-6)
    assert decayed(0.8, 60) == pytest.approx(0.8 * math.exp(-2.0), rel=1e-6)


def test_mail_decay_never_grows_and_never_goes_negative():
    previous = decayed(0.9, 0)
    for age in (1, 7, 30, 90, 365, 3650):
        current = decayed(0.9, age)
        assert 0.0 <= current <= previous
        previous = current


def test_a_recent_weaker_match_outranks_an_old_strong_one():
    # Three months is four half-lives. "The budget email" almost always means
    # the recent one.
    recent = decayed(0.70, 2)
    old = decayed(0.85, 90)
    assert recent > old


def test_decay_does_not_simply_sort_by_date():
    # A much better match five days old still beats a weak one from yesterday.
    # Decay is a tilt, not an override — otherwise search would be a mailbox
    # sorted by received_at.
    strong_and_older = decayed(0.95, 5)
    weak_and_newer = decayed(0.70, 1)
    assert strong_and_older > weak_and_newer


def test_events_lean_the_other_way():
    # A meeting tomorrow matters more than the identical meeting last month.
    # Applying mail's decay to a calendar would rank a finished event above the
    # one the user is about to walk into.
    assert boosted(0.70, 1) > boosted(0.70, -1)
    assert boosted(0.70, 1) >= 0.70
    assert decayed(0.70, 1) < 0.70


def test_a_soon_event_outranks_a_slightly_better_distant_one():
    tomorrow = boosted(0.70, 1)
    three_weeks_ago = boosted(0.75, -21)
    assert tomorrow > three_weeks_ago


def test_the_boost_fades_with_distance():
    # Next week beats next quarter. Something eight months out is barely more
    # relevant than something in the past.
    assert boosted(0.70, 3) >= boosted(0.70, 30) >= boosted(0.70, 240)


def test_ordering_a_mixed_mail_list_end_to_end():
    mail = [
        {"id": "old_strong", "cn": 0.88, "age_days": 120},
        {"id": "recent_ok", "cn": 0.72, "age_days": 3},
        {"id": "recent_weak", "cn": 0.58, "age_days": 1},
    ]
    ranked = sorted(mail, key=lambda m: -decayed(m["cn"], m["age_days"]))
    assert [m["id"] for m in ranked] == ["recent_ok", "recent_weak", "old_strong"]

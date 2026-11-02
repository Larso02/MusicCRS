"""The feedback-parsing evaluation (scripts/eval_feedback.py): its checker and the integrity of its three sets."""

import pytest

ev = pytest.importorskip("scripts.eval_feedback")


def test_the_three_sets_have_thirty_distinct_sentences_each_and_do_not_overlap():
    sentences = {name: [s.lower() for s, _ in cases] for name, cases in ev.SETS.items()}
    assert set(sentences) == {"tuned", "heldout", "fresh"}
    for name, items in sentences.items():
        assert len(items) == 30 and len(set(items)) == 30, name
    everything = [s for items in sentences.values() for s in items]
    assert len(set(everything)) == 90                         # the held-out and fresh sets share nothing with the tuned one


def test_positions_must_be_on_the_right_side():
    out = {"action": "feedback", "liked_positions": [1, 2], "disliked_positions": [3]}
    assert ev.holds(out, {"liked_positions": {1, 2}})
    assert ev.holds(out, {"liked_positions": {1}, "disliked_positions": {3}})
    assert not ev.holds(out, {"liked_positions": {3}})        # 3 is on the other side
    assert not ev.holds(out, {"disliked_positions": {1, 3}})


def test_names_must_not_appear_on_the_opposite_side():
    out = {"liked_artists": ["Coldplay"], "disliked_artists": ["Linkin Park"]}
    assert ev.holds(out, {"liked_artists": ["coldplay"], "disliked_artists": ["linkin"]})
    assert not ev.holds(out, {"liked_artists": ["linkin"]})
    assert not ev.holds({"liked_artists": ["Coldplay"], "disliked_artists": ["Coldplay"]}, {"liked_artists": ["coldplay"]})


def test_wants_compares_word_stems_and_looks_at_wishes_hints_and_liked_tags():
    assert ev.mentions("something heavier", "heavy") and ev.mentions("more upbeat", "upbeat")
    assert not ev.mentions("something calm", "heavy")
    assert ev.holds({"wish": "something heavier"}, {"wants": "heavy"})
    assert ev.holds({"action": "recommend", "hint": "more upbeat"}, {"wants": "upbeat"})
    assert ev.holds({"liked_tags": ["alternative rock"]}, {"wants": "alternative rock"})
    assert not ev.holds({"wish": "something calm"}, {"wants": "upbeat"})


def test_a_select_counts_as_liking_the_songs():
    assert ev.holds({"action": "select", "positions": [3, 4]}, {"liked_positions": {3, 4}})
    assert ev.holds({"action": "select", "mode": "all"}, {"liked_positions": {1, 2, 3, 4, 5}})
    assert ev.holds({"action": "select", "except": [2]}, {"liked_positions": {1, 3, 4, 5}})
    assert not ev.holds({"action": "select", "positions": [3]}, {"disliked_positions": {3}})
    assert ev.holds({"action": "select", "mode": "none"}, {"selects": set()})


def test_any_of_accepts_any_listed_reading():
    expect = {"any_of": [{"disliked_artists": ["linkin"]}, {"disliked_positions": {4}}]}
    assert ev.holds({"disliked_artists": ["Linkin Park"]}, expect)
    assert ev.holds({"disliked_positions": [4]}, expect)
    assert not ev.holds({"disliked_positions": [2]}, expect)


def test_variety_must_match_exactly():
    assert ev.holds({"variety": "more"}, {"variety": "more"})
    assert not ev.holds({"variety": "less"}, {"variety": "more"})
    assert not ev.holds({}, {"variety": "more"})

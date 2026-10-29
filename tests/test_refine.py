"""Part 4: preference profile, Rocchio, feedback reranker, MMR, explanations and the chat flow."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from musiccrs.track_database import TrackDatabase, artist_key, match_tag

refine_module = pytest.importorskip("musiccrs.refine")
user_profile = pytest.importorskip("musiccrs.user_profile")
Profile = user_profile.Profile
Config = refine_module.Config


# ---------- the profile ----------
def test_a_new_profile_is_empty():
    p = Profile()
    assert p.is_empty() and p.summary()["liked_tags"] == []


def test_the_newest_opinion_wins():
    p = Profile()
    p.like_tag("metal"); p.dislike_tag("metal")
    p.like_artist("coldplay"); p.dislike_artist("coldplay")
    p.like_track("t1"); p.dislike_track("t1"); p.dislike_track("t1")
    assert p.disliked_tags == {"metal"} and not p.liked_tags
    assert p.disliked_artists == {"coldplay"} and not p.liked_artists
    assert p.disliked_tracks == ["t1"] and p.liked_tracks == []
    p.like_track("t1")
    assert p.liked_tracks == ["t1"] and p.disliked_tracks == []


def test_wishes_are_kept_once_and_clear_resets_everything():
    p = Profile()
    p.add_wish(" more upbeat "); p.add_wish("more upbeat"); p.add_wish("  ")
    assert p.wishes == ["more upbeat"] and not p.is_empty()
    p.like_tag("rock")
    p.clear()
    assert p.is_empty() and p.wishes == [] and p.liked_tags == set()


# ---------- the engine ----------
@pytest.fixture(scope="module")
def ix(agent_module):
    return agent_module.get_term_index(TrackDatabase())


def idx_of(ix, title, artist):
    for i in range(ix.n):
        if ix.titles[i] == title and artist in ix.artist_names[i]:
            return i
    raise LookupError((title, artist))


def test_the_index_covers_the_catalog(ix):
    assert ix.n == len(ix.ids) == 47071
    assert ix.bm25_weights.shape == (ix.n, len(ix.vocab)) == ix.unit.shape
    assert np.allclose(np.asarray(ix.unit.multiply(ix.unit).sum(axis=1)).ravel()[:100], 1.0, atol=1e-4)


def test_text_vector_keeps_known_words_and_sums_to_one(ix):
    q = ix.text_vector("grunge rock zzzzqqq")
    assert q.nnz == 2 and abs(q.sum() - 1.0) < 1e-6
    assert ix.text_vector("zzzzqqq").nnz == 0


def test_rocchio_moves_the_query_towards_liked_and_away_from_disliked(ix):
    liked = [idx_of(ix, "Yellow", "Coldplay")]
    disliked = [idx_of(ix, "Smells Like Teen Spirit", "Nirvana")]
    q = ix.text_vector("rock")
    new = ix.rocchio(q, liked, disliked, 1.0, 0.7, 0.3, top_terms=40)
    dense, old = np.asarray(new.todense()).ravel(), np.asarray(q.todense()).ravel()
    assert dense.min() >= 0                                       # BM25 cannot use negative weights
    assert np.count_nonzero(dense) <= 40
    coldplay_term = ix.vocab["coldplay"]
    assert dense[coldplay_term] > old[coldplay_term]              # a term of the liked track gained weight
    nirvana_only = ix.vocab["nirvana"]
    assert dense[nirvana_only] == 0                               # a term of the disliked track is gone


def test_without_feedback_rocchio_leaves_the_query_alone(ix):
    q = ix.text_vector("rock")
    same = refine_module.refine(ix, "rock", [], Profile(), Config(rocchio=True, rerank=True), k=10)
    base = refine_module.refine(ix, "rock", [], Profile(), Config(rocchio=False, rerank=False), k=10)
    assert [i.idx for i in same] == [i.idx for i in base] and q.nnz == 1


def test_played_and_disliked_tracks_never_come_back(ix):
    played = [idx_of(ix, "Yellow", "Coldplay")]
    p = Profile()
    nirvana = idx_of(ix, "Lithium", "Nirvana")
    p.dislike_track(ix.ids[nirvana])
    items = refine_module.refine(ix, "rock", played, p, Config(), k=50)
    got = {i.idx for i in items}
    assert played[0] not in got and nirvana not in got and len(got) == 50
    again = refine_module.refine(ix, "rock", played, p, Config(), k=50)
    assert [i.idx for i in again] == [i.idx for i in items]       # deterministic


def test_exclude_keeps_earlier_suggestions_out(ix):
    first = refine_module.refine(ix, "rock", [], Profile(), Config(), k=5)
    more = refine_module.refine(ix, "rock", [], Profile(), Config(), k=5, exclude=[i.idx for i in first])
    assert not {i.idx for i in first} & {i.idx for i in more}


def top_share(ix, items, tag):
    return float(np.mean([tag in ix.desc_tags[i.idx] for i in items[:5]]))


def test_a_rejected_tag_leaves_the_top_five(ix):
    played = [idx_of(ix, "Lithium", "Nirvana")]
    base_cfg = Config(rocchio=False, rerank=False)
    base = refine_module.refine(ix, "grunge", played, Profile(), base_cfg, k=20)
    tag = "grunge"
    assert top_share(ix, base, tag) >= 0.6                         # the scenario: grunge dominates
    p = Profile(); p.dislike_tag(tag)
    after = refine_module.refine(ix, "grunge", played, p, Config(), k=20)
    assert top_share(ix, after, tag) < top_share(ix, base, tag) - 0.3     # cut by at least 30 points
    assert all(i.parts["rejected"] == 0 for i in after[:5] if tag not in ix.desc_tags[i.idx])


def test_a_rejected_artist_is_penalised(ix):
    played = [idx_of(ix, "Lithium", "Nirvana")]
    p = Profile(); p.dislike_artist(artist_key("Nirvana"))
    items = refine_module.refine(ix, "grunge", played, p, Config(), k=10)
    assert sum("Nirvana" in ix.artist_names[i.idx] for i in items[:5]) <= 1


def test_a_liked_artist_and_tag_are_rewarded(ix):
    p = Profile(); p.like_artist(artist_key("Coldplay"))
    items = refine_module.refine(ix, "music", [], p, Config(), k=5)
    assert sum("Coldplay" in ix.artist_names[i.idx] for i in items) >= 3
    assert all(i.parts["liked_artist"] > 0 for i in items if "Coldplay" in ix.artist_names[i.idx])


def test_liked_tracks_pull_the_results_their_way(ix):
    liked = [idx_of(ix, "Yellow", "Coldplay"), idx_of(ix, "Clocks", "Coldplay")]
    p = Profile()
    for i in liked:
        p.like_track(ix.ids[i])
    with_rocchio = refine_module.refine(ix, "songs", [], p, Config(rerank=False), k=10)
    without = refine_module.refine(ix, "songs", [], p, Config(rocchio=False, rerank=False), k=10)
    gain = lambda items: sum("Coldplay" in ix.artist_names[i.idx] for i in items)
    assert gain(with_rocchio) > gain(without)


def pairwise_similarity(ix, items):
    m = ix.unit[[i.idx for i in items]]
    sims = (m @ m.T).toarray()
    return (sims.sum() - len(items)) / (len(items) * (len(items) - 1))


def test_mmr_makes_the_list_more_varied_and_lambda_one_changes_nothing(ix):
    played = [idx_of(ix, "Lithium", "Nirvana")]
    plain = refine_module.refine(ix, "rock", played, Profile(), Config(), k=5)
    varied = refine_module.refine(ix, "rock", played, Profile(), Config(mmr_lambda=0.4), k=5)
    same = refine_module.refine(ix, "rock", played, Profile(), Config(mmr_lambda=1.0), k=5)
    assert pairwise_similarity(ix, varied) < pairwise_similarity(ix, plain)
    assert [i.idx for i in same] == [i.idx for i in plain]


def test_explanations_come_from_the_score_components(ix):
    played = [idx_of(ix, "Lithium", "Nirvana")]
    p = Profile(); p.like_tag("grunge"); p.like_artist(artist_key("Pearl Jam"))
    items = refine_module.refine(ix, "grunge", played, p, Config(), k=40)
    texts = [refine_module.explain(ix, i) for i in items]
    assert any("same artist as a song on your playlist" in t for t in texts)
    assert any("grunge" in t for t in texts)
    assert all(t for t in texts)
    pearl_jam = [refine_module.explain(ix, i) for i in items if "Pearl Jam" in ix.artist_names[i.idx]]
    assert pearl_jam and all("by an artist you liked" in t for t in pearl_jam)


def test_a_rejected_attribute_is_named_in_the_explanation(ix):
    item = refine_module.Item(0, 1.0, {"bm25": 0.2, "rejected": -2.0, "rejected_hit": ["metal"], "liked_tags_hit": []})
    assert "metal" in refine_module.explain(ix, item) and "rejected" in refine_module.explain(ix, item)


def test_diff_lists():
    d = refine_module.diff_lists([1, 2, 3], [2, 3, 4])
    assert d == {"added": [4], "removed": [1], "kept": [2, 3]}


@pytest.mark.parametrize("text, expected", [
    ("grunge", "grunge"), ("Alternative Rock", "alternative rock"), ("alternative rok", "alternative rock"),
    ("90s vibe", "90s"), ("heavy", "heavy metal"), ("zzzz qqqq", None), ("", None),
])
def test_match_tag(text, expected):
    got = match_tag(text)
    if expected == "heavy metal":
        assert got is not None and "heavy" in got            # any tag that contains the word
    else:
        assert got == expected


# ---------- the chat flow ----------
class FakeLLM:
    """Router replies come from the script; the one-sentence summary can be switched on."""

    def __init__(self, script, phrase=None):
        self.script, self.phrase, self.calls = script, phrase, []
        self.chat = self
        self.completions = self

    def with_options(self, **options):
        return self

    def create(self, model, messages, temperature, max_tokens):
        content = messages[-1]["content"]
        self.calls.append(content)
        if content.startswith("You are MusicCRS"):
            if self.phrase is None:
                raise TimeoutError("no phrasing")
            reply = self.phrase
        else:
            reply = json.dumps(self.script[content.split("MESSAGE: ")[-1]])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


@pytest.fixture()
def chat(agent, say, agent_module):
    from musiccrs.router import Router

    def _chat(message, action, phrase=None):
        fake = FakeLLM({message: action}, phrase)
        agent._llm = fake
        agent._router = Router(fake, "m")
        return say(agent, message)

    return _chat


@pytest.fixture()
def with_list(agent, say, chat):
    say(agent, "add Yellow by Coldplay")
    chat("recommend something", {"action": "recommend"})
    return agent


def test_feedback_on_numbered_songs(with_list, chat):
    agent = with_list
    liked, disliked = agent._last_recs[0], agent._last_recs[3]
    text, _ = chat("I like the first one but not the fourth", {
        "action": "feedback", "liked_positions": [1], "disliked_positions": [4]})
    assert liked["track_id"] in agent._profile.liked_tracks
    assert disliked["track_id"] in agent._profile.disliked_tracks
    assert disliked["track_id"] not in [t["track_id"] for t in agent._last_recs]
    assert "Got it" in text and "Compared with the last list" in text


def test_feedback_about_styles_and_artists(with_list, chat):
    agent = with_list
    text, _ = chat("less coldplay more grunge", {
        "action": "feedback", "disliked_artists": ["Coldplay"], "liked_tags": ["grunge"]})
    assert "coldplay" in agent._profile.disliked_artists and "grunge" in agent._profile.liked_tags
    assert "more grunge" in text and "Coldplay" in text
    assert not any("Coldplay" in t["artist_name"] for t in agent._last_recs)


def test_an_unknown_style_becomes_a_wish_and_an_unknown_complaint_is_ignored(with_list, chat):
    agent = with_list
    chat("more zorbo please", {"action": "feedback", "liked_tags": ["zorbo"], "disliked_tags": ["qwxz"]})
    assert agent._profile.wishes == ["zorbo"] and not agent._profile.disliked_tags


def test_feedback_without_anything_usable_falls_back_to_the_rules(with_list, chat, agent):
    text, _ = chat("hmm", {"action": "feedback"})
    assert "Here is the new list" not in text and agent._profile.is_empty()


def test_the_llm_only_words_the_summary(with_list, chat):
    agent = with_list
    text, _ = chat("no more coldplay", {"action": "feedback", "disliked_artists": ["Coldplay"]},
                   phrase="Sure, I will leave Coldplay out.")
    assert "Sure, I will leave Coldplay out." in text
    assert "coldplay" in agent._profile.disliked_artists          # the facts came from the action, not the sentence


def test_make_it_more_varied(with_list, chat):
    agent = with_list
    chat("make it more varied", {"action": "feedback", "variety": "more"})
    assert agent._variety == 0.5
    chat("more of the same", {"action": "recommend", "variety": "less"})
    assert agent._variety == 1.0


def test_a_hint_with_recommend_is_a_stated_wish(with_list, chat):
    agent = with_list
    chat("something more mellow", {"action": "recommend", "hint": "more mellow"})
    assert agent._profile.wishes == ["more mellow"]


def test_adding_a_suggested_song_likes_it_and_removing_it_rejects_it(with_list, chat, say, agent):
    first = agent._last_recs[0]["track_id"]
    say(agent, "1")                                               # picks the first suggestion
    assert first in agent._playlist and first in agent._profile.liked_tracks
    say(agent, f"/remove {first}")
    assert first not in agent._playlist
    assert first in agent._profile.disliked_tracks and first not in agent._profile.liked_tracks


def test_songs_that_were_never_suggested_are_not_feedback(agent, say):
    say(agent, "add Clocks by Coldplay")
    track_id = agent._playlist[0]
    say(agent, f"/remove {track_id}")
    assert agent._profile.is_empty()


def test_clearing_the_playlist_forgets_the_preferences(with_list, chat, say, agent):
    chat("no more coldplay", {"action": "feedback", "disliked_artists": ["Coldplay"]})
    assert not agent._profile.is_empty()
    say(agent, "/clear")
    assert agent._profile.is_empty() and agent._last_recs == []


def test_recommend_more_after_feedback_does_not_repeat_suggestions(with_list, chat, agent):
    chat("no more coldplay", {"action": "feedback", "disliked_artists": ["Coldplay"]})
    first = {t["track_id"] for t in agent._last_recs}
    chat("more", {"action": "recommend"})
    assert not first & {t["track_id"] for t in agent._last_recs}


def test_the_router_sees_the_last_recommendations(with_list, chat, agent):
    chat("pick one", {"action": "select", "positions": [1]})       # closes the list
    chat("number 2 was bad", {"action": "feedback", "disliked_positions": [2]})
    prompt = next(c for c in agent._router.llm.calls if "MESSAGE: number 2 was bad" in c)
    assert "LAST RECOMMENDATIONS: 1. " in prompt

"""R8: playlist generation from a description (LLM and retrievers replaced by stubs)."""

import numpy as np
import pytest

from musiccrs.track_database import TrackDatabase


@pytest.fixture(autouse=True)
def _needs_confirm_flow(agent):
    if not hasattr(agent, "_handle_confirmation"):
        pytest.skip("the replace-playlist confirmation is not implemented yet")


class StubGenerator:
    """Stands in for PlaylistGenerator: no LLM call, no retrieval."""

    def __init__(self, track_ids):
        self.track_ids = track_ids
        self.requests = []

    def generate(self, description):
        self.requests.append(description)
        return {"track_ids": self.track_ids, "actual_minutes": 61.5}


@pytest.fixture()
def gen_agent(agent):
    db = TrackDatabase()
    ids = [db.search_tracks(t, a, limit=1)[0]["track_id"] for t, a in [
        ("Yellow", "Coldplay"), ("Heart-Shaped Box", "Nirvana"), ("Clocks", "Coldplay")]]
    agent._llm = object()                       # "an LLM is configured"
    agent._playlist_generator = StubGenerator(ids)
    return agent


def test_without_an_llm_generation_is_refused(agent, say):
    agent._llm = None
    text, tracks = say(agent, "make me a playlist for the gym")
    assert "requires an LLM" in text and tracks == []


@pytest.mark.parametrize("text, wanted", [
    ("/generate energetic music for the gym", "energetic music for the gym"),
    ("make me a playlist for studying", "studying"),
    ("Make a playlist of 80s rock", "80s rock"),
    ("create a playlist with sad love songs", "sad love songs"),
    ("can you build me a playlist for a road trip?", "a road trip"),
    ("please generate a playlist about summer", "summer"),
    ("give me a playlist for working out", "working out"),
    ("put together a playlist for a party", "a party"),
    ("/generate", ""),
    ("make a playlist", ""),
])
def test_generate_requests_are_recognised(agent_module, text, wanted):
    m = agent_module._GENERATE_RE.match(text)
    assert m is not None and m.group("d") == wanted


@pytest.mark.parametrize("text", [
    "add Yellow by Coldplay", "show playlist", "recommend something",
    "what is the best playlist", "I love this playlist", "clear playlist",
])
def test_other_messages_are_not_generate_requests(agent_module, text):
    assert not agent_module._GENERATE_RE.match(text)


def test_empty_playlist_is_generated_without_asking(gen_agent, say):
    text, tracks = say(gen_agent, "make me a playlist for the gym")
    assert "Created a new playlist" in text and "3 songs" in text and "62 minutes" in text
    assert len(tracks) == 3
    assert gen_agent._playlist_generator.requests == ["the gym"]


def test_missing_description_explains_usage(gen_agent, say):
    text, tracks = say(gen_agent, "/generate")
    assert "describe" in text.lower() and tracks == []
    assert gen_agent._playlist_generator.requests == []


def test_non_empty_playlist_asks_before_replacing(gen_agent, say):
    say(gen_agent, "add Yellow by Coldplay")
    text, tracks = say(gen_agent, "/generate calm piano music")
    assert "replaces your current playlist" in text and "(1 songs)" in text
    assert len(tracks) == 1                                   # nothing changed yet
    assert gen_agent._playlist_generator.requests == []       # and no LLM call was made


@pytest.mark.parametrize("yes", ["yes", "Yes!", "yeah", "ok", "sure", "go ahead", "yes please"])
def test_yes_replaces_the_playlist(gen_agent, say, yes):
    say(gen_agent, "add Yellow by Coldplay")
    say(gen_agent, "/generate calm piano music")
    text, tracks = say(gen_agent, yes)
    assert "Created a new playlist" in text and "calm piano music" in text
    assert len(tracks) == 3
    assert gen_agent._playlist_generator.requests == ["calm piano music"]


@pytest.mark.parametrize("no", ["no", "No thanks", "cancel", "nope", "keep it", "don't"])
def test_no_keeps_the_playlist(gen_agent, say, no):
    say(gen_agent, "add Yellow by Coldplay")
    say(gen_agent, "/generate calm piano music")
    text, tracks = say(gen_agent, no)
    assert "keep your current playlist" in text
    assert len(tracks) == 1
    assert gen_agent._playlist_generator.requests == []


def test_any_other_message_cancels_the_question(gen_agent, say):
    say(gen_agent, "add Yellow by Coldplay")
    say(gen_agent, "/generate calm piano music")
    text, tracks = say(gen_agent, "show playlist")            # handled normally
    assert "Yellow" in text and len(tracks) == 1
    text, tracks = say(gen_agent, "yes")                      # the question is gone
    assert "Created a new playlist" not in text
    assert len(tracks) == 1
    assert gen_agent._playlist_generator.requests == []


def test_yes_without_a_question_does_nothing_special(gen_agent, say):
    text, tracks = say(gen_agent, "yes")
    assert "Created a new playlist" not in text and tracks == []


def test_generating_resets_earlier_recommendations(gen_agent, say):
    say(gen_agent, "add Yellow by Coldplay")
    say(gen_agent, "recommend something")
    assert gen_agent._recommended
    say(gen_agent, "/generate calm piano music")
    say(gen_agent, "yes")
    assert gen_agent._recommended == []


def test_generating_closes_an_open_choice_list(gen_agent, say):
    say(gen_agent, "add Yellow by Coldplay")
    say(gen_agent, "recommend something")                      # an open "pick a number" list
    text, _ = say(gen_agent, "make me a playlist for the gym")  # a command: moves on, asks
    assert "replaces your current playlist" in text
    text, tracks = say(gen_agent, "yes")
    assert "Created a new playlist" in text and len(tracks) == 3
    reply, _ = say(gen_agent, "2")                            # no stale choice list
    assert not reply.startswith("Added")


def test_failure_keeps_the_playlist(gen_agent, say):
    def boom(description):
        raise RuntimeError("LLM down")

    gen_agent._playlist_generator.generate = boom
    say(gen_agent, "add Yellow by Coldplay")
    say(gen_agent, "/generate calm piano music")
    text, tracks = say(gen_agent, "yes")
    assert "couldn't generate" in text
    assert len(tracks) == 1


def test_unknown_track_ids_from_the_generator_are_dropped(gen_agent, say):
    gen_agent._playlist_generator.track_ids = ["not-a-real-id"]
    text, tracks = say(gen_agent, "/generate calm piano music")
    assert "couldn't find suitable songs" in text and tracks == []


def test_help_lists_playlist_generation(agent, say):
    text, _ = say(agent, "what can you do?")
    assert "Generate a playlist" in text
    text, _ = say(agent, "how do I create a playlist?")
    assert "make me a playlist" in text


# ---------- retrieval pieces ----------
@pytest.fixture()
def pg():
    return pytest.importorskip("musiccrs.playlist_generator")


def test_rrf_rewards_agreement_between_rankings(pg):
    fused = pg.rrf([["a", "b", "c"], ["b", "d", "a"]])
    assert fused[0] == "b" and fused[1] == "a"          # in both lists, high in both
    assert set(fused) == {"a", "b", "c", "d"}            # union, no duplicates


def test_rrf_weights_and_single_ranking(pg):
    assert pg.rrf([["a", "b"]]) == ["a", "b"]
    assert pg.rrf([["a", "b"], ["b", "a"]], weights=[1, 3])[0] == "b"
    assert pg.rrf([]) == []


def test_dense_index_returns_nearest_tracks_first(pg):
    ids = ["t0", "t1", "t2", "t3"]
    matrix = np.array([[1, 0], [0, 1], [0.7071, 0.7071], [-1, 0]], dtype=np.float32)
    index = pg.DenseMetaIndex(ids, matrix=matrix, encoder=lambda text: np.array([1.0, 0.0]))
    assert index.search("anything", topk=3) == ["t0", "t2", "t1"]


class FailingIndex:
    failed = False

    def search(self, query, topk):
        raise RuntimeError("no model")


def test_generation_falls_back_to_bm25_when_dense_fails(pg):
    generator = pg.PlaylistGenerator.__new__(pg.PlaylistGenerator)
    generator.retriever = type("R", (), {"text_to_item_retrieval": lambda self, q, topk: ["a", "b"]})()
    generator.dense = FailingIndex()
    ids, method = generator.candidates("rock")
    assert ids == ["a", "b"] and method == "bm25"
    assert generator.dense.failed                      # not retried on every request


def test_generation_fuses_bm25_and_dense(pg):
    class Dense:
        failed = False

        def search(self, query, topk):
            return ["b", "c"]

    generator = pg.PlaylistGenerator.__new__(pg.PlaylistGenerator)
    generator.retriever = type("R", (), {"text_to_item_retrieval": lambda self, q, topk: ["a", "b"]})()
    generator.dense = Dense()
    ids, method = generator.candidates("rock")
    assert method == "bm25+dense" and ids[0] == "b" and set(ids) == {"a", "b", "c"}

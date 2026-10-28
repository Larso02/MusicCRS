"""Tags are free text from Last.fm users. The LLM judged once which describe the music;
the program has no hand-written stop list or filter patterns."""

import json
from types import SimpleNamespace

import pytest

import musiccrs.track_database as track_database
from musiccrs.track_database import TrackDatabase

pytestmark = pytest.mark.skipif(
    not hasattr(track_database, "is_descriptive_tag"), reason="LLM tag labels are not integrated yet"
)

LABELS = track_database.DATABASE_PATH.parent / "tag_labels.json"
needs_labels = pytest.mark.skipif(not LABELS.exists(), reason="run: python -m scripts.label_tags")


def test_there_is_no_hand_written_filter():
    assert not hasattr(track_database, "TAG_STOPLIST")
    assert not hasattr(track_database, "_OPINION")


@needs_labels
def test_the_label_file_is_complete_and_well_formed():
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    labels = data["labels"]
    assert data["model"] and data["min_tracks"] >= 1
    assert len(labels) > 10_000
    assert set(labels.values()) <= {0, 1}
    kept = sum(labels.values())
    assert 0.1 * len(labels) < kept < 0.7 * len(labels)      # neither "everything" nor "nothing"


@needs_labels
@pytest.mark.parametrize("tag", ["rock", "alternative rock", "indie", "female vocalists", "hip-hop", "80s", "chill", "mellow", "sad", "instrumental"])
def test_descriptive_tags_are_kept(tag):
    assert track_database.is_descriptive_tag(tag)
    assert track_database.is_descriptive_tag(tag.upper())      # case does not matter


@needs_labels
@pytest.mark.parametrize("tag", ["favorites", "seen live", "awesome", "albums i own", "2013", "10 of 10 stars", "", "   "])
def test_noise_tags_are_dropped(tag):
    assert not track_database.is_descriptive_tag(tag)


@needs_labels
def test_tags_nobody_judged_are_dropped():
    assert not track_database.is_descriptive_tag("a tag nobody has ever typed 12345")


def test_the_name_of_the_artist_or_song_is_not_a_style():
    assert not track_database.is_descriptive_tag("rock", names={"rock"})


def test_without_the_label_file_every_tag_passes(monkeypatch):
    monkeypatch.setattr(track_database, "_descriptive_tags", lambda: None)
    assert track_database.is_descriptive_tag("whatever you like")
    assert not track_database.is_descriptive_tag("")


@needs_labels
def test_clean_tags_dedupes_limits_and_orders_by_how_common_they_are():
    counts = {"rock": 900, "indie": 100, "pop": 400}
    out = track_database.clean_tags(["Indie", "indie", "favorites", "pop", "rock", "2010"], k=2, counts=counts)
    assert out == ["rock", "pop"]
    assert track_database.clean_tags(["Rock", "rock", "favorites", "Indie"]) == ["rock", "indie"]


@pytest.fixture(scope="module")
def db():
    return TrackDatabase()


@needs_labels
def test_song_tags_are_descriptive(db):
    song = db.get_details(db.search_tracks("Yellow", "Coldplay", limit=1)[0]["track_id"])
    tags = db.song_tags(song)
    assert 1 <= len(tags) <= 6
    assert "coldplay" not in tags and "yellow" not in tags
    assert all(track_database.is_descriptive_tag(t) for t in tags)


@needs_labels
def test_artist_tags_are_descriptive(db):
    key = db.find_artists("Coldplay", limit=1)[0]["key"]
    tags = db.artist_tags(key)
    assert tags and "coldplay" not in tags
    assert all(track_database.is_descriptive_tag(t) for t in tags)
    assert any(t in tags for t in ("rock", "alternative", "alternative rock", "pop"))


@needs_labels
def test_answers_use_clean_tags(agent, say):
    text, _ = say(agent, "what genre is Yellow by Coldplay?")
    low = text.lower()
    assert "tagged" in low and "favo" not in low and "seen live" not in low and "awesome" not in low


@needs_labels
def test_the_recommender_vocabulary_is_the_judged_tags_only(agent_module):
    rec = agent_module.get_recommender(TrackDatabase())
    assert 1000 < len(rec.vocab) < 6000                 # was about 164,000 raw tags
    assert all(track_database.is_descriptive_tag(t) for t in rec.vocab)


@needs_labels
def test_recommendation_reasons_show_clean_tags(agent, say):
    say(agent, "add Yellow by Coldplay")
    text, _ = say(agent, "recommend something")
    for noise in ("favo", "seen live", "memories", "awesome", "love it"):
        assert noise not in text.lower()


# ---------- the labelling script ----------
@pytest.fixture()
def label_tags():
    return pytest.importorskip("scripts.label_tags")


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.chat = self
        self.completions = self

    def create(self, model, temperature, messages):
        self.prompts.append(messages[0]["content"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def test_judge_maps_numbers_back_to_tags(label_tags):
    client = FakeClient(["Here you go: [1, 3]"])
    assert label_tags.judge(client, ["rock", "favorites", "indie"]) == {"rock", "indie"}
    assert "1. rock" in client.prompts[0] and "3. indie" in client.prompts[0]


def test_judge_ignores_numbers_outside_the_list_and_accepts_an_empty_answer(label_tags):
    assert label_tags.judge(FakeClient(["[2, 9, 0]"]), ["a", "b"]) == {"b"}
    assert label_tags.judge(FakeClient(["[]"]), ["a", "b"]) == set()


def test_judge_retries_and_then_gives_up(label_tags, monkeypatch):
    monkeypatch.setattr(label_tags.time, "sleep", lambda s: None)
    assert label_tags.judge(FakeClient([TimeoutError("slow"), "no list", "[1]"]), ["rock"]) == {"rock"}
    assert label_tags.judge(FakeClient(["?", "?", "?"]), ["rock"]) is None

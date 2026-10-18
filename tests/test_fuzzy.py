"""Tests for the typo-tolerant track search (musiccrs/fuzzy.py).

These tests are skipped until musiccrs/fuzzy.py exists.
Run from the repo root: python -m pytest tests
"""

import time

import pytest

from musiccrs.track_database import DATABASE_PATH, TrackDatabase

fuzzy = pytest.importorskip(
    "musiccrs.fuzzy", reason="typo-tolerant search not added yet"
)
pytestmark = pytest.mark.skipif(
    not DATABASE_PATH.is_file(), reason="run `python -m scripts.import_tracks` first"
)


@pytest.fixture(scope="module")
def db():
    return TrackDatabase()


@pytest.fixture(scope="module")
def index(db):
    return fuzzy.FuzzyTrackIndex(db)


def best(index, db, text):
    track_id, similarity = index.search(text, k=1)[0]
    return db.get_track(track_id), similarity


# ---------- the search itself ----------
@pytest.mark.parametrize(
    "query, title, artist",
    [
        ("heartshaped box by nirvana", "Heart-Shaped Box", "Nirvana"),
        ("yelow by coldplay", "Yellow", "Coldplay"),        # letter missing
        ("yellow by coldpay", "Yellow", "Coldplay"),        # typo in the artist
        ("coldplay yellow", "Yellow", "Coldplay"),          # artist first, no separator
        ("nirvana heartshaped box", "Heart-Shaped Box", "Nirvana"),
        ("stnad by me", "Stand By Me", None),               # "by" is part of the title
        ("heartshaped box", "Heart-Shaped Box", None),      # missing space
    ],
)
def test_typos_find_the_right_song(index, db, query, title, artist):
    track, _ = best(index, db, query)
    assert track["track_name"].lower() == title.lower()
    if artist:
        assert artist in track["artist_name"]


def test_nonsense_has_low_similarity(index, db):
    _, similarity = best(index, db, "asdfghjkl qwerty zzzz")
    assert similarity < 0.6          # below the 'offer a guess' floor


def test_empty_queries(index):
    assert index.search("") == []
    assert index.search("!!!") == []


def test_search_is_fast(index):
    start = time.time()
    for _ in range(10):
        index.search("heartshaped box by nirvana")
    assert (time.time() - start) / 10 < 0.1       # well inside the 3-5 s budget


# ---------- through the agent ----------
def test_typo_with_artist_is_accepted_and_explained(agent, say):
    text, tracks = say(agent, "add yelow by coldplay")
    assert text.startswith("Added") and "I read" in text
    assert [t["track_name"] for t in tracks] == ["Yellow"]


def test_title_containing_by_with_a_typo(agent, say):
    text, tracks = say(agent, "add stnad by me")
    assert text.startswith("Added")
    assert tracks[0]["track_name"].lower() == "stand by me"


def test_uncertain_typo_asks_instead_of_adding(agent, say):
    text, tracks = say(agent, "add roes")
    assert "Did you mean" in text and tracks == []


def test_nonsense_is_not_guessed(agent, say):
    text, tracks = say(agent, "add asdfghjkl qwerty zzzz")
    assert "couldn't find" in text and tracks == []


def test_exact_requests_are_not_annotated(agent, say):
    text, _ = say(agent, "add Yellow by Coldplay")
    assert text.startswith("Added") and "I read" not in text

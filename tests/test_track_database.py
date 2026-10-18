"""Tests for the SQLite track catalog (musiccrs/track_database.py).

Needs the database: run `python -m scripts.import_tracks` once first.
Run from the repo root: python -m pytest tests
"""

import sqlite3
from contextlib import closing

import pytest

from musiccrs.track_database import DATABASE_PATH, TrackDatabase, norm

needs_db = pytest.mark.skipif(
    not DATABASE_PATH.is_file(),
    reason="run `python -m scripts.import_tracks` first",
)


# ---------- norm(): no database needed ----------
def test_norm_punctuation_and_case():
    assert norm("Don't Stop Believin'") == "dont stop believin"
    assert norm("Heart-Shaped Box") == "heart shaped box"


def test_norm_accents_and_case():
    assert norm("Se Preparó") == norm("SE PREPARÓ") == "se preparo"


def test_norm_removes_like_wildcards():
    assert "%" not in norm("100% Pure") and "_" not in norm("a_b")
    assert norm("%") == "" and norm("_") == ""


# ---------- database ----------
@pytest.fixture(scope="module")
def db():
    return TrackDatabase()


@needs_db
def test_schema_and_size():
    with closing(sqlite3.connect(DATABASE_PATH)) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tracks)")}
        count = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    assert {"track_id", "title_norm", "artist_norm", "popularity"} <= cols
    assert count == 47071


@needs_db
def test_title_by_artist(db):
    hits = db.find_tracks("Yellow by Coldplay")
    assert hits and hits[0]["exact"]
    assert hits[0]["track_name"] == "Yellow"
    assert "Coldplay" in hits[0]["artist_name"]


@needs_db
def test_artist_colon_title(db):
    hits = db.find_tracks("Coldplay: Yellow")
    assert hits and hits[0]["track_name"] == "Yellow"
    assert "Coldplay" in hits[0]["artist_name"]


@needs_db
def test_title_only_is_exact(db):
    hits = db.search_tracks("Yellow")
    assert hits and hits[0]["exact"]


@needs_db
def test_loose_punctuation_and_spacing(db):
    assert db.find_tracks("heart shaped box")[0]["track_name"] == "Heart-Shaped Box"


@needs_db
def test_accents_are_case_insensitive(db):
    hits = db.find_tracks("SE PREPARÓ")
    assert hits and hits[0]["track_name"] == "Se Preparó"


@needs_db
def test_title_containing_by_is_not_split(db):
    hits = db.find_tracks("Stand By Me")
    assert hits and hits[0]["exact"]
    assert hits[0]["track_name"].lower() == "stand by me"


@needs_db
def test_unknown_track(db):
    assert db.find_tracks("asdfghjkl qwerty zzzz") == []


@needs_db
def test_wildcards_are_harmless(db):
    assert db.find_tracks("%") == []
    assert db.find_tracks("_") == []


@needs_db
def test_limit_is_respected(db):
    assert len(db.find_tracks("love", limit=5)) <= 5


@needs_db
def test_exact_matches_come_first_and_by_popularity(db):
    hits = db.find_tracks("Love", limit=10)
    flags = [h["exact"] for h in hits]
    assert flags == sorted(flags, reverse=True)       # all exact before partial
    exact_pop = [h["popularity"] for h in hits if h["exact"]]
    assert exact_pop == sorted(exact_pop, reverse=True)


@needs_db
def test_get_tracks_keeps_order_and_skips_unknown(db):
    a = db.find_tracks("Yellow by Coldplay")[0]["track_id"]
    b = db.find_tracks("Iris by The Goo Goo Dolls")[0]["track_id"]
    got = db.get_tracks([b, "no-such-id", a])
    assert [t["track_id"] for t in got] == [b, a]
    assert db.get_tracks([]) == []


@needs_db
def test_get_track(db):
    track = db.find_tracks("Yellow by Coldplay")[0]
    assert db.get_track(track["track_id"])["track_name"] == "Yellow"
    assert db.get_track("no-such-id") is None

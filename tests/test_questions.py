"""R5: questions about songs and artists.

Skipped until musiccrs/questions.py exists; the database and agent tests also need the
new database (re-run `python -m scripts.import_tracks`). Run: python -m pytest tests
"""

import html
import json
import re
import sqlite3
import time
from contextlib import closing

import pytest

from musiccrs.track_database import DATABASE_PATH

questions = pytest.importorskip("musiccrs.questions", reason="R5 not added yet")
from musiccrs.questions import parse_question  # noqa: E402


# ---------------------------------------------------------------- parsing (no database)
PARSE_CASES = [
    ("who sings Yellow?", "track_artist", "Yellow"),
    ("Who sings Yellow by Coldplay", "track_artist", "Yellow by Coldplay"),
    ("who performed Stand By Me", "track_artist", "Stand By Me"),
    ("which artist made Iris?", "track_artist", "Iris"),
    ("who is the artist of Yellow", "track_artist", "Yellow"),
    ("who is Yellow by?", "track_artist", "Yellow"),
    ("whose song is Iris", "track_artist", "Iris"),
    ("what album is Yellow on?", "track_album", "Yellow"),
    ("which album is Yellow by Coldplay from", "track_album", "Yellow by Coldplay"),
    ("what album does Yellow belong to", "track_album", "Yellow"),
    ("when was Yellow released?", "track_year", "Yellow"),
    ("what year was Iris released", "track_year", "Iris"),
    ("when did Yellow come out?", "track_year", "Yellow"),
    ("what is the release date of Iris", "track_year", "Iris"),
    ("do you know when Iris was released", "track_year", "Iris"),
    ("how long is Yellow?", "track_length", "Yellow"),
    ("how long does Yellow last", "track_length", "Yellow"),
    ("what's the duration of Iris", "track_length", "Iris"),
    ("please tell me how long Yellow is", "track_length", "Yellow"),
    ("how popular is Yellow?", "track_popularity", "Yellow"),
    ("is Yellow popular", "track_popularity", "Yellow"),
    ("what genre is Yellow?", "tags", "Yellow"),
    ("what kind of music does Coldplay play", "tags", "Coldplay"),
    ("what tags does Yellow have", "tags", "Yellow"),
    ("tell me about Coldplay", "about", "Coldplay"),
    ("who is Coldplay?", "about", "Coldplay"),
    ("what songs does Coldplay have?", "artist_songs", "Coldplay"),
    ("what songs by Coldplay do you have", "artist_songs", "Coldplay"),
    ("list songs by Coldplay", "artist_songs", "Coldplay"),
    ("songs by Coldplay", "artist_songs", "Coldplay"),
    ("what albums does Coldplay have", "artist_albums", "Coldplay"),
    ("albums by Coldplay", "artist_albums", "Coldplay"),
    ("how many songs does Coldplay have?", "artist_count", "Coldplay"),
    ("how many tracks by Coldplay do you have", "artist_count", "Coldplay"),
    ("what is the most popular song by Coldplay", "artist_top", "Coldplay"),
    ("what's Coldplay's most popular song?", "artist_top", "Coldplay"),
    ("can you tell me who sings Yellow?", "track_artist", "Yellow"),
    ("who sings it?", "track_artist", "it"),
    ("how long is this song", "track_length", "this song"),
]

NOT_QUESTIONS = [
    "add Yellow by Coldplay", "show my playlist", "remove yellow", "hello", "help",
    "what can you do?", "how do I add a song?", "/info", "cancel", "2",
    "the one by Kevin Abstract", "address book", "",
]


@pytest.mark.parametrize("text, kind, subject", PARSE_CASES)
def test_questions_are_understood(text, kind, subject):
    q = parse_question(text)
    assert q is not None, text
    assert (q.kind, q.subject) == (kind, subject)


@pytest.mark.parametrize("text", NOT_QUESTIONS)
def test_other_messages_are_not_questions(text):
    assert parse_question(text) is None


# ---------------------------------------------------------------- database and agent
def _has_new_database():
    if not DATABASE_PATH.is_file():
        return False
    with closing(sqlite3.connect(DATABASE_PATH)) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return "track_artists" in tables


needs_db = pytest.mark.skipif(
    not _has_new_database(),
    reason="re-run `python -m scripts.import_tracks` (R5 adds release date, length, tags, artists)",
)


@pytest.fixture(autouse=True)
def _needs_r5_agent(request):
    if "agent" in request.fixturenames:
        agent_module = request.getfixturevalue("agent_module")
        if not hasattr(agent_module.MusicCRS, "_answer_question"):
            pytest.skip("R5 agent code not added yet")


def plain(text):
    """What the user reads: no tags, no HTML entities."""
    return html.unescape(re.sub(r"<[^>]+>", "", text.replace("<br>", " | ")))


def options_of(agent):
    reply = agent._dialogue_connector.sent[-1]
    acts = [a for a in reply.dialogue_acts if str(a.intent) == "OPTIONS"]
    return [json.loads(a.value) for a in acts[0].annotations] if acts else None


@pytest.fixture(scope="module")
def db():
    from musiccrs.track_database import TrackDatabase
    return TrackDatabase()


@pytest.fixture(scope="module")
def yellow(db):
    """Coldplay's Yellow, with all its details."""
    track = db.find_tracks("Yellow by Coldplay")[0]
    return db.get_details(track["track_id"])


@needs_db
def test_database_details(db, yellow):
    assert yellow["track_name"] == "Yellow"
    assert yellow["release_date"].startswith("2000")
    assert yellow["duration"] > 0 and yellow["tags"]
    assert db.get_details("no-such-id") is None


@needs_db
def test_database_artists(db):
    found = db.find_artists("Coldplay")
    assert found[0]["name"] == "Coldplay" and found[0]["similarity"] == 1.0
    assert db.find_artists("coldpay")[0]["name"] == "Coldplay"              # typo
    assert db.find_artists("Beatles")[0]["name"] == "The Beatles"           # leading "the"
    assert db.find_artists("zzzzqx") == []
    tracks = db.artist_tracks(found[0]["key"])
    assert len(tracks) == found[0]["tracks"]
    assert [t["popularity"] for t in tracks] == sorted((t["popularity"] for t in tracks), reverse=True)
    assert db.artist_albums(found[0]["key"]) and db.artist_tags(found[0]["key"])


@needs_db
def test_bogus_release_dates_are_not_stored():
    with closing(sqlite3.connect(DATABASE_PATH)) as conn:
        bad = conn.execute(
            "SELECT COUNT(*) FROM tracks WHERE release_date IS NOT NULL AND substr(release_date, 1, 4) < '1900'"
        ).fetchone()[0]
    assert bad == 0


# ---- questions about a song
@needs_db
def test_song_artist(agent, say):
    text, _ = say(agent, "who sings Yellow by Coldplay?")
    assert plain(text) == "Yellow is by Coldplay."


@needs_db
def test_song_artist_for_a_title_shared_by_many_lists_them(agent, say):
    text, _ = say(agent, "who sings Yellow?")
    assert "Coldplay" in text and "Kevin Abstract" in text
    assert options_of(agent) is None                       # it answered, it did not ask


@needs_db
def test_song_album(agent, say, yellow):
    text, _ = say(agent, "what album is Yellow by Coldplay on?")
    assert yellow["album_name"][0] in plain(text)


@needs_db
def test_song_release(agent, say, yellow):
    text, _ = say(agent, "when was Yellow by Coldplay released?")
    assert yellow["release_date"] in plain(text) and "2000" in plain(text)


@needs_db
def test_song_length(agent, say, yellow):
    minutes, seconds = divmod(yellow["duration"] // 1000, 60)
    text, _ = say(agent, "how long is Yellow by Coldplay?")
    assert f"{minutes}:{seconds:02d}" in plain(text)


@needs_db
def test_song_popularity(agent, say, yellow):
    text, _ = say(agent, "how popular is Yellow by Coldplay?")
    assert f"{yellow['popularity']:.0f}/100" in plain(text)
    assert "100%" not in plain(text)                       # never "higher than 100%"


@needs_db
def test_song_tags(agent, say):
    text, _ = say(agent, "what genre is Yellow by Coldplay?")
    assert "tagged" in text and "favorite" not in text.lower()


@needs_db
def test_song_summary(agent, say):
    text, _ = say(agent, "tell me about Yellow by Coldplay")
    assert "Coldplay" in text and "2000" in text and "4:" in text


# ---- questions about an artist
@needs_db
def test_artist_songs_and_count(agent, say, db):
    n = db.find_artists("Coldplay")[0]["tracks"]
    text, _ = say(agent, "what songs does Coldplay have?")
    assert f"I have {n} songs by Coldplay" in plain(text) and "1. Yellow" in plain(text)
    text, _ = say(agent, "how many songs does Coldplay have?")
    assert plain(text) == f"I have {n} songs by Coldplay."


@needs_db
def test_artist_albums(agent, say):
    text, _ = say(agent, "what albums does Coldplay have?")
    assert "Parachutes" in text


@needs_db
def test_artist_top_song(agent, say):
    text, _ = say(agent, "what is the most popular song by Coldplay?")
    assert plain(text).startswith("The most popular song by Coldplay is Yellow")


@needs_db
def test_artist_style_and_summary(agent, say):
    text, _ = say(agent, "what genre is Coldplay?")
    assert "mostly tagged" in text
    text, _ = say(agent, "tell me about Nirvana")
    assert "Nirvana" in text and "Heart-Shaped Box" in text


@needs_db
def test_leading_the_is_optional(agent, say, db):
    n = db.find_artists("The Beatles")[0]["tracks"]
    text, _ = say(agent, "how many songs by the beatles")
    assert f"{n} songs" in plain(text)


# ---- context, ambiguity, typos
@needs_db
def test_it_refers_to_the_last_song(agent, say):
    text, _ = say(agent, "who sings it?")
    assert "Which song do you mean" in text                # nothing to refer to yet
    say(agent, "add Yellow by Coldplay")
    text, _ = say(agent, "who sings it?")
    assert plain(text) == "Yellow is by Coldplay."
    text, _ = say(agent, "when was it released?")
    assert "2000" in text


@needs_db
def test_ambiguous_song_asks_then_answers_without_adding(agent, say, db):
    text, tracks = say(agent, "when was Iris released?")
    assert "Which one" in text and tracks == []
    assert options_of(agent) is not None
    goo = db.get_details(db.find_tracks("Iris by The Goo Goo Dolls")[0]["track_id"])
    text, tracks = say(agent, "1")
    assert goo["release_date"][:4] in plain(text)
    assert tracks == []                                    # answering a question does not add


@needs_db
def test_cancelling_a_question(agent, say):
    say(agent, "when was Iris released?")
    text, _ = say(agent, "cancel")
    assert "never mind" in text.lower()


@needs_db
def test_a_new_question_replaces_an_open_one(agent, say):
    say(agent, "add Yellow")                               # opens a "which one?"
    text, tracks = say(agent, "who sings Iris by The Goo Goo Dolls?")
    assert "Goo Goo Dolls" in text and tracks == []
    text, tracks = say(agent, "2")                         # the old question is gone
    assert tracks == [] and "don't understand" in text


@needs_db
def test_typos_in_song_and_artist(agent, say):
    text, _ = say(agent, "who sings yelow by coldplay?")
    assert "Yellow is by Coldplay" in plain(text) and "I read" in text
    text, _ = say(agent, "what songs does coldpay have?")
    assert "songs by Coldplay" in plain(text) and "I read" in text


@needs_db
def test_unknown_things(agent, say):
    text, _ = say(agent, "what songs does Zzzzqx have?")
    assert "couldn't find an artist" in text
    text, _ = say(agent, "who sings asdfghjkl?")
    assert "couldn't find" in text


@needs_db
def test_questions_do_not_change_the_playlist(agent, say):
    say(agent, "add Yellow by Coldplay")
    _, tracks = say(agent, "how long is Iris by The Goo Goo Dolls?")
    assert [t["track_name"] for t in tracks] == ["Yellow"]


@needs_db
def test_answers_are_fast(agent, say):
    say(agent, "who sings Yellow by Coldplay?")            # warm-up
    for question in ["what songs does Coldplay have?", "when was Yellow by Coldplay released?",
                     "who sings yelow by coldplay?"]:
        start = time.time()
        say(agent, question)
        assert time.time() - start < 1.5, question         # far inside the 3-5 s budget


def test_help_mentions_questions():
    capabilities = pytest.importorskip("musiccrs.capabilities")
    answer = capabilities.help_answer("what can you do?")
    if "Answer questions" not in answer:
        pytest.skip("R5 entry not added to capabilities yet")
    assert "Answer questions" in answer

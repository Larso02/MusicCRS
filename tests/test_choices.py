"""R4: asking the user which song they mean, with ranked, clickable options.

Skipped until the agent has the R4 code (_ask_which). Run: python -m pytest tests
"""

import json
import sqlite3
from contextlib import closing

import pytest

from musiccrs.track_database import DATABASE_PATH

pytestmark = pytest.mark.skipif(
    not DATABASE_PATH.is_file(), reason="run `python -m scripts.import_tracks` first"
)


@pytest.fixture(autouse=True)
def _needs_r4(agent_module):
    if not hasattr(agent_module.MusicCRS, "_ask_which"):
        pytest.skip("R4 not added yet")


def options_of(agent):
    """The OPTIONS dialogue act of the last reply, as a list of {title, payload}."""
    reply = agent._dialogue_connector.sent[-1]
    acts = [a for a in reply.dialogue_acts if str(a.intent) == "OPTIONS"]
    return [json.loads(a.value) for a in acts[0].annotations] if acts else None


def artists_of(tracks):
    return [", ".join(t["artist_name"]) for t in tracks]


def test_ambiguous_title_asks_and_does_not_add(agent, say):
    text, tracks = say(agent, "add Yellow")
    assert "Which one" in text and tracks == []
    options = options_of(agent)
    assert 3 <= len(options) <= 6                      # up to 5 songs + Cancel
    assert options[-1] == {"title": "Cancel", "payload": "cancel"}
    assert options[0]["title"] == "Yellow by Coldplay"  # most popular first


def test_options_are_shown_numbered_in_the_text(agent, say):
    text, _ = say(agent, "add Yellow")
    assert "1. Yellow" in text and "2. Yellow" in text


@pytest.mark.parametrize("reply", ["2", "2nd", "the second one", "option 2", "second"])
def test_pick_by_number_or_ordinal(agent, say, reply):
    say(agent, "add Yellow")
    second = options_of(agent)[1]["title"]
    text, tracks = say(agent, reply)
    assert text.startswith("Added") and len(tracks) == 1
    assert f"{tracks[0]['track_name']} by {', '.join(tracks[0]['artist_name'])}" == second


def test_pick_the_last_one(agent, say):
    say(agent, "add Yellow")
    last_song = options_of(agent)[-2]["title"]         # [-1] is Cancel
    _, tracks = say(agent, "the last one")
    assert f"{tracks[0]['track_name']} by {', '.join(tracks[0]['artist_name'])}" == last_song


def test_pick_by_artist_name(agent, say):
    say(agent, "add Yellow")
    _, tracks = say(agent, "the one by Kevin Abstract")
    assert [", ".join(t["artist_name"]) for t in tracks] == ["Kevin Abstract"]


def test_clicking_a_button_sends_its_label(agent, say):
    say(agent, "add Yellow")
    payload = options_of(agent)[2]["payload"]
    text, tracks = say(agent, payload)
    assert text.startswith("Added") and len(tracks) == 1


@pytest.mark.parametrize("reply", ["cancel", "never mind", "none of them", "no"])
def test_cancel(agent, say, reply):
    say(agent, "add Yellow")
    text, tracks = say(agent, reply)
    assert "won't add" in text and tracks == []
    assert options_of(agent) is None


def test_unclear_reply_keeps_asking(agent, say):
    say(agent, "add Yellow")
    text, tracks = say(agent, "banana")
    assert "didn't catch" in text and tracks == []
    assert options_of(agent) is not None               # the buttons come back


def test_a_new_command_drops_the_question(agent, say):
    say(agent, "add Yellow")
    text, _ = say(agent, "show my playlist")
    assert "empty" in text
    text, tracks = say(agent, "2")                     # no longer an answer
    assert tracks == [] and "don't understand" in text


def test_specifying_the_artist_avoids_the_question(agent, say):
    text, tracks = say(agent, "add Yellow by Coldplay")
    assert text.startswith("Added") and len(tracks) == 1
    assert options_of(agent) is None


def test_options_are_ranked_by_the_playlist(agent, say):
    # Put another Kevin Abstract song on the playlist; his "Yellow" should now come first.
    with closing(sqlite3.connect(DATABASE_PATH)) as conn:
        row = conn.execute(
            "SELECT track_id FROM tracks WHERE artist_norm = 'kevin abstract' "
            "AND title_norm != 'yellow' ORDER BY popularity DESC LIMIT 1"
        ).fetchone()
    assert row, "catalog should have another Kevin Abstract song"
    say(agent, f"add {row[0]}")
    say(agent, "add Yellow")
    assert options_of(agent)[0]["title"] == "Yellow by Kevin Abstract"


def test_uncertain_typo_offers_buttons_and_picking_adds(agent, say):
    text, tracks = say(agent, "add roes")
    assert "Did you mean" in text and tracks == []
    assert options_of(agent) is not None
    text, tracks = say(agent, "1")
    assert text.startswith("Added") and len(tracks) == 1


def test_already_on_the_playlist(agent, say):
    say(agent, "add Yellow")
    say(agent, "1")
    say(agent, "add Yellow")
    text, tracks = say(agent, "1")
    assert "already in your playlist" in text and len(tracks) == 1

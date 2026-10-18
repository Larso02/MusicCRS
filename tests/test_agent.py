"""Conversation-level tests for the MusicCRS agent (no LLM, real catalog).

Run from the repo root: python -m pytest tests
"""

import pytest

from musiccrs.track_database import DATABASE_PATH

pytestmark = pytest.mark.skipif(
    not DATABASE_PATH.is_file(), reason="run `python -m scripts.import_tracks` first"
)


def test_every_reply_carries_the_playlist_state(agent, say):
    for message in ["hello there", "show my playlist", "/info", "add Yellow by Coldplay"]:
        _, tracks = say(agent, message)
        assert tracks is not None, f"no PLAYLIST_STATE act for {message!r}"


def test_add_view_remove_clear(agent, say):
    text, tracks = say(agent, "add Yellow by Coldplay")
    assert text.startswith("Added") and [t["track_name"] for t in tracks] == ["Yellow"]

    text, tracks = say(agent, "Please add the song Iris by The Goo Goo Dolls to my playlist")
    assert text.startswith("Added") and len(tracks) == 2

    text, tracks = say(agent, "show my playlist")
    assert "2 songs" in text and len(tracks) == 2

    text, tracks = say(agent, "remove yellow from my playlist")
    assert text.startswith("Removed") and [t["track_name"] for t in tracks] == ["Iris"]

    text, tracks = say(agent, "/clear")
    assert "cleared" in text.lower() and tracks == []

    text, _ = say(agent, "show my playlist")
    assert text == "Your playlist is empty."


def test_no_duplicates(agent, say):
    say(agent, "add Yellow by Coldplay")
    text, tracks = say(agent, "add Yellow by Coldplay")
    assert "already in your playlist" in text and len(tracks) == 1


def test_only_catalog_tracks_can_be_added(agent, say):
    text, tracks = say(agent, "add asdfghjkl qwerty zzzz")
    assert "couldn't find" in text and tracks == []


def test_remove_by_id_like_the_ui_button(agent, say):
    _, tracks = say(agent, "add Yellow by Coldplay")
    text, tracks = say(agent, f"/remove {tracks[0]['track_id']}")
    assert text.startswith("Removed") and tracks == []


def test_remove_unknown_keeps_playlist(agent, say):
    say(agent, "add Yellow by Coldplay")
    text, tracks = say(agent, "remove nothing here")
    assert "Could not find" in text and len(tracks) == 1


def test_simulator_style_commands(agent, say):
    _, tracks = say(agent, "/add Coldplay: Yellow")
    assert [t["track_name"] for t in tracks] == ["Yellow"]
    _, tracks = say(agent, "/playlist")
    assert len(tracks) == 1
    _, tracks = say(agent, "/remove Coldplay: Yellow")
    assert tracks == []


def test_track_labels_are_html_escaped(agent):
    label = agent._track_label({"track_name": "A & B", "artist_name": ["<x>"]})
    assert "A &amp; B" in label and "&lt;x&gt;" in label
    assert "<x>" not in label


def test_words_like_address_are_not_commands(agent, say):
    text, tracks = say(agent, "address book")
    assert "don't understand" in text and tracks == []

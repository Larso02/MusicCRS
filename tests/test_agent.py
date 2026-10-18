"""Conversation-level tests for the MusicCRS agent (no LLM, real catalog).

Run from the repo root: python -m pytest tests
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import musiccrs.track_database as track_database
from musiccrs.track_database import DATABASE_PATH

pytestmark = pytest.mark.skipif(
    not DATABASE_PATH.is_file(), reason="run `python -m scripts.import_tracks` first"
)

ROOT = Path(__file__).resolve().parents[1]


def _load_agent_module():
    """musiccrs.py does `from track_database import ...`; alias it, then load by path."""
    sys.modules.setdefault("track_database", track_database)
    spec = importlib.util.spec_from_file_location(
        "musiccrs_agent", ROOT / "musiccrs" / "musiccrs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeConnector:
    def __init__(self):
        self.sent = []

    def register_agent_utterance(self, utterance):
        self.sent.append(utterance)


@pytest.fixture()
def agent():
    module = _load_agent_module()
    bot = module.MusicCRS(use_llm=False)
    bot._dialogue_connector = FakeConnector()
    return bot


def say(bot, text):
    """Send a user message; return (reply text, playlist tracks from the dialogue act)."""
    bot._dialogue_connector.sent.clear()
    bot.receive_utterance(SimpleNamespace(text=text))
    reply = bot._dialogue_connector.sent[-1]
    acts = [a for a in reply.dialogue_acts if str(a.intent) == "PLAYLIST_STATE"]
    tracks = [json.loads(ann.value) for ann in acts[0].annotations] if acts else None
    return reply.text, tracks


def test_every_reply_carries_the_playlist_state(agent):
    for message in ["hello there", "show my playlist", "/info", "add Yellow by Coldplay"]:
        _, tracks = say(agent, message)
        assert tracks is not None, f"no PLAYLIST_STATE act for {message!r}"


def test_add_view_remove_clear(agent):
    text, tracks = say(agent, "add Yellow by Coldplay")
    assert text.startswith("Added") and [t["track_name"] for t in tracks] == ["Yellow"]

    text, tracks = say(agent, "Please add the song Iris to my playlist")
    assert text.startswith("Added") and len(tracks) == 2

    text, tracks = say(agent, "show my playlist")
    assert "2 songs" in text and len(tracks) == 2

    text, tracks = say(agent, "remove yellow from my playlist")
    assert text.startswith("Removed") and [t["track_name"] for t in tracks] == ["Iris"]

    text, tracks = say(agent, "/clear")
    assert "cleared" in text.lower() and tracks == []

    text, _ = say(agent, "show my playlist")
    assert text == "Your playlist is empty."


def test_no_duplicates(agent):
    say(agent, "add Yellow by Coldplay")
    text, tracks = say(agent, "add Yellow by Coldplay")
    assert "already in your playlist" in text and len(tracks) == 1


def test_only_catalog_tracks_can_be_added(agent):
    text, tracks = say(agent, "add asdfghjkl qwerty zzzz")
    assert "couldn't find" in text and tracks == []


def test_remove_by_id_like_the_ui_button(agent):
    _, tracks = say(agent, "add Yellow by Coldplay")
    text, tracks = say(agent, f"/remove {tracks[0]['track_id']}")
    assert text.startswith("Removed") and tracks == []


def test_remove_unknown_keeps_playlist(agent):
    say(agent, "add Yellow by Coldplay")
    text, tracks = say(agent, "remove nothing here")
    assert "Could not find" in text and len(tracks) == 1


def test_simulator_style_commands(agent):
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


def test_words_like_address_are_not_commands(agent):
    text, tracks = say(agent, "address book")
    assert "don't understand" in text and tracks == []

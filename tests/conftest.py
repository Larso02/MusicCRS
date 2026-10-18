"""Shared test setup: importable repo root, plus a no-LLM agent with a fake connector."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeConnector:
    def __init__(self):
        self.sent = []

    def register_agent_utterance(self, utterance):
        self.sent.append(utterance)


@pytest.fixture(scope="session")
def agent_module():
    """Load musiccrs/musiccrs.py once per test run (the fuzzy index is built once)."""
    import musiccrs.track_database as track_database

    # musiccrs.py uses top-level imports (`from track_database import ...`)
    sys.modules.setdefault("track_database", track_database)
    try:
        import musiccrs.fuzzy as fuzzy

        sys.modules.setdefault("fuzzy", fuzzy)
    except ImportError:
        pass                               # typo-tolerant search not added yet

    spec = importlib.util.spec_from_file_location(
        "musiccrs_agent", ROOT / "musiccrs" / "musiccrs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def agent(agent_module):
    bot = agent_module.MusicCRS(use_llm=False)
    bot._dialogue_connector = FakeConnector()
    return bot


@pytest.fixture()
def say():
    def _say(bot, text):
        """Send a user message; return (reply text, playlist tracks from the dialogue act)."""
        bot._dialogue_connector.sent.clear()
        bot.receive_utterance(SimpleNamespace(text=text))
        reply = bot._dialogue_connector.sent[-1]
        acts = [a for a in reply.dialogue_acts if str(a.intent) == "PLAYLIST_STATE"]
        tracks = [json.loads(a.value) for a in acts[0].annotations] if acts else None
        return reply.text, tracks

    return _say

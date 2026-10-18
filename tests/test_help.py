"""R3: learning about the system's functionality in natural language.

Skipped until musiccrs/capabilities.py exists. Run: python -m pytest tests
"""

import pytest

pytest.importorskip("musiccrs.capabilities", reason="R3 not added yet")
from musiccrs.capabilities import help_answer, is_help_request  # noqa: E402


@pytest.mark.parametrize(
    "text",
    [
        "help",
        "/help",
        "Help me",
        "what can you do?",
        "How do I add a song?",
        "how can I remove a song",
        "what commands are there",
        "what are your features",
        "how does this work",
    ],
)
def test_help_requests_are_recognised(text):
    assert is_help_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "add Yellow by Coldplay",
        "show my playlist",
        "remove yellow",
        "clear my playlist",
        "hello",
        "Yellow by Coldplay",
        "address book",
    ],
)
def test_other_messages_are_not_help_requests(text):
    assert not is_help_request(text)


def test_overview_lists_what_is_available():
    answer = help_answer("what can you do?")
    for title in ("Add songs", "Remove songs", "Show the playlist", "Clear the playlist"):
        assert title in answer


def test_topic_questions_get_a_focused_answer():
    add = help_answer("how do I add a song?")
    assert "Add songs" in add and "Remove songs" not in add
    assert "Remove songs" in help_answer("how do I remove a song?")


def test_unimplemented_features_are_not_advertised():
    assert "Recommend" not in help_answer("what can you do?")


# ---------- through the agent ----------
def test_agent_answers_help_questions(agent, say):
    text, tracks = say(agent, "what can you do?")
    assert "Add songs" in text and tracks is not None
    text, _ = say(agent, "how do I remove a song?")
    assert "Remove songs" in text


def test_help_does_not_block_adding_a_song_called_help(agent, say):
    text, _ = say(agent, "add Help by The Beatles")
    assert "what I can do" not in text           # treated as an add, not as help


def test_unknown_messages_point_to_help(agent, say):
    text, _ = say(agent, "blah blah")
    assert "help" in text.lower()

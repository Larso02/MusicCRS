"""Part 2 requirements checked end to end: wording variants for R2/R4/R7 found in the demo run."""

import re

import pytest

from musiccrs.track_database import TrackDatabase


@pytest.fixture(autouse=True)
def _needs_fixes(agent_module):
    if not hasattr(agent_module, "_CLEAR_RE"):
        pytest.skip("the natural-language view/clear patterns are not implemented yet")


def plain(text):
    return re.sub(r"<[^>]+>", " ", text.replace("<br>", "\n"))


# ---------- R2: view and clear in natural language ----------
@pytest.mark.parametrize("text", [
    "show playlist", "show my playlist", "view playlist", "view my playlist", "show me my playlist",
    "show me the playlist", "display my playlist", "what's in my playlist", "what is in my playlist",
    "what is on my playlist", "what songs do I have", "what songs are in my playlist",
    "list my songs", "show me my songs", "can you show my playlist?", "my playlist", "playlist",
    "/playlist", "/list", "please show the playlist",
])
def test_view_phrasings(agent_module, text):
    assert agent_module._VIEW_RE.match(text)


@pytest.mark.parametrize("text", [
    "clear", "clear playlist", "clear my playlist", "clear the playlist", "clear it", "clear everything",
    "empty my playlist", "empty the playlist", "reset the playlist", "start over", "start again",
    "remove everything", "delete all songs", "remove all the songs from my playlist", "please clear my playlist",
    "can you clear the playlist?", "wipe my playlist", "/clear",
])
def test_clear_phrasings(agent_module, text):
    assert agent_module._CLEAR_RE.match(text)


@pytest.mark.parametrize("text", [
    "add Yellow", "remove Yellow", "clear skies by Oasis", "show me Coldplay songs", "who sings clear",
    "what songs does Coldplay have", "recommend something", "start over by Dolly Parton",
])
def test_other_messages_are_not_view_or_clear(agent_module, text):
    assert not agent_module._VIEW_RE.match(text) and not agent_module._CLEAR_RE.match(text)


def test_clear_and_view_through_the_agent(agent, say):
    say(agent, "add Yellow by Coldplay")
    text, tracks = say(agent, "what songs do I have?")
    assert "Yellow" in text and len(tracks) == 1
    text, tracks = say(agent, "start over")
    assert "cleared" in text and tracks == []


# ---------- R2: removing loosely ----------
@pytest.fixture()
def listed(agent, say):
    for q in ["Yellow by Coldplay", "Clocks by Coldplay", "Smells Like Teen Spirit by Nirvana"]:
        say(agent, f"add {q}")
    return agent


@pytest.mark.parametrize("command", [
    "remove the song by nirvana", "remove nirvana", "remove the nirvana song", "remove smells like teen spirit",
    "take out the track by Nirvana", "delete the song smells like teen spirit by nirvana",
])
def test_remove_by_artist_or_title(listed, say, command):
    text, tracks = say(listed, command)
    assert text.startswith("Removed") and len(tracks) == 2
    assert all("Nirvana" not in ", ".join(t["artist_name"]) for t in tracks)


def test_removing_an_ambiguous_artist_asks_which_one(listed, say):
    text, tracks = say(listed, "remove the song by coldplay")
    assert "Which one should I remove" in text and len(tracks) == 3
    assert "Yellow" in text and "Clocks" in text
    text, tracks = say(listed, "2")
    assert text.startswith("Removed") and len(tracks) == 2
    assert "Clocks" not in [t["track_name"] for t in tracks]


def test_cancelling_a_removal_keeps_everything(listed, say):
    say(listed, "remove coldplay")
    text, tracks = say(listed, "cancel")
    assert "won't remove" in text and len(tracks) == 3


def test_removing_something_that_is_not_there(listed, say):
    text, tracks = say(listed, "remove the song by metallica")
    assert "Could not find" in text and len(tracks) == 3


# ---------- R7: the assignment's examples ----------
@pytest.fixture()
def offer(agent):
    db = TrackDatabase()
    wanted = [("Yellow", "Coldplay"), ("Clocks", "Coldplay"), ("Smells Like Teen Spirit", "Nirvana"),
              ("Heart-Shaped Box", "Nirvana"), ("My Hero", "Foo Fighters")]
    tracks = [db.search_tracks(t, a, limit=1)[0] for t, a in wanted]
    agent._ask_which("Based on your playlist, you might like these:", tracks)
    return tracks


@pytest.mark.parametrize("command, positions", [
    ("add them except the one by Nirvana", [0, 1, 4]),
    ("add them all except the Nirvana ones", [0, 1, 4]),
    ("add everything but the song by Foo Fighters", [0, 1, 2, 3]),
    ("add all except the coldplay songs", [2, 3, 4]),
    ("all but the first two", [2, 3, 4]),
    ("add the first two", [0, 1]),
    ("add all", [0, 1, 2, 3, 4]),
    ("add the last one", [4]),
])
def test_selection_examples_from_the_assignment(agent, say, offer, command, positions):
    text, tracks = say(agent, command)
    assert [t["track_id"] for t in tracks] == [offer[i]["track_id"] for i in positions]


@pytest.mark.parametrize("command", ["none", "nothing", "nothing, thanks", "none of them", "no songs", "no thanks", "cancel"])
def test_adding_no_songs_at_all(agent, say, offer, command):
    text, tracks = say(agent, command)
    assert tracks == [] and "won't add" in text
    assert agent._pending is None


def test_a_normal_add_still_drops_an_open_list(agent, say, offer):
    text, tracks = say(agent, "add Clocks by Coldplay")      # names a song: not a selection
    assert text.startswith("Added") and len(tracks) == 1


# ---------- R8: the replace question ----------
def test_replace_question_has_correct_grammar(agent, say):
    agent._llm = object()
    say(agent, "add Yellow by Coldplay")
    text, _ = say(agent, "make me a playlist for the gym")
    assert "(1 song)" in text
    say(agent, "no")
    say(agent, "add Clocks by Coldplay")
    text, _ = say(agent, "make me a playlist for the gym")
    assert "(2 songs)" in text

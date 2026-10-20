"""R7: picking several songs from a list in plain English."""

import re

import pytest

from musiccrs.track_database import TrackDatabase

selection = pytest.importorskip("musiccrs.selection")
parse_selection = selection.parse_selection

LABELS = ["yellow coldplay", "clocks coldplay", "smells like teen spirit nirvana",
          "heart shaped box nirvana", "my hero foo fighters"]
ALL5 = [1, 2, 3, 4, 5]


# ---------- the parser ----------
@pytest.mark.parametrize("text, want", [
    ("2", [2]), ("1 and 3", [1, 3]), ("add 1 and 3", [1, 3]), ("1, 2, 5", [1, 2, 5]), ("1 3", [1, 3]),
    ("number 2 and number 4", [2, 4]), ("can you add number 2 and number 4", [2, 4]),
    ("the first and the last", [1, 5]), ("first, third and fifth", [1, 3, 5]),
    ("2nd and 4th", [2, 4]), ("the second one", [2]), ("just the second", [2]), ("only 1 and 2", [1, 2]),
    ("I'd like the first and third one please", [1, 3]),
    ("the first two", [1, 2]), ("first 3", [1, 2, 3]), ("top three please", [1, 2, 3]),
    ("the last two", [4, 5]), ("last 3 songs", [3, 4, 5]), ("last", [5]), ("the last one", [5]),
    ("all", ALL5), ("all of them", ALL5), ("add everything", ALL5), ("add them all", ALL5),
    ("2-4", [2, 3, 4]), ("2 - 4", [2, 3, 4]), ("1 to 3", [1, 2, 3]), ("first through third", [1, 2, 3]),
    ("all except 2", [1, 3, 4, 5]), ("all but the second", [1, 3, 4, 5]),
    ("everything except the last", [1, 2, 3, 4]), ("1-4 except 2", [1, 3, 4]),
    ("all but 1 and 2", [3, 4, 5]), ("not the first", [2, 3, 4, 5]), ("2 but not 3", [2]),
    ("1 and 3 without 3", [1]),
])
def test_numbers_and_quantities(text, want):
    assert parse_selection(text, 5) == want


@pytest.mark.parametrize("text", [
    "", "hello", "yes", "no", "cancel", "none of them", "neither", "both", "7", "0", "1 and 9", "first 9",
    "remove Clocks", "show playlist", "clear playlist", "recommend more", "who sings it",
    "add Yellow by Coldplay", "the coldplay ones", "all except", "except",
])
def test_things_that_are_not_selections(text):
    assert parse_selection(text, 5) is None


def test_both_only_means_something_for_two_items():
    assert parse_selection("both", 2) == [1, 2]
    assert parse_selection("add both", 2) == [1, 2]
    assert parse_selection("both", 3) is None


@pytest.mark.parametrize("text, want", [
    ("the coldplay ones", [1, 2]), ("the nirvana songs", [3, 4]), ("coldplay", [1, 2]),
    ("foo fighters", [5]), ("the first and the nirvana ones", [1, 3, 4]),
    ("all except the coldplay ones", [3, 4, 5]), ("add Yellow by Coldplay", [1]),
])
def test_picking_by_name(text, want):
    assert parse_selection(text, 5, LABELS) == want


def test_unknown_names_select_nothing():
    assert parse_selection("the beatles ones", 5, LABELS) is None


# ---------- through the agent ----------
@pytest.fixture(autouse=True)
def _needs_r7(agent):
    if not hasattr(agent, "_add_many"):
        pytest.skip("R7 is not implemented yet")


def plain(text):
    return re.sub(r"<[^>]+>", " ", text.replace("<br>", "\n"))


@pytest.fixture()
def offer(agent):
    """An open list of five songs whose answer adds them (what 'recommend' leaves behind)."""
    db = TrackDatabase()
    wanted = [("Yellow", "Coldplay"), ("Clocks", "Coldplay"), ("Smells Like Teen Spirit", "Nirvana"),
              ("Heart-Shaped Box", "Nirvana"), ("My Hero", "Foo Fighters")]
    tracks = [db.search_tracks(t, a, limit=1)[0] for t, a in wanted]
    agent._ask_which("Based on your playlist, you might like these:", tracks)
    return tracks


def ids(tracks):
    return [t["track_id"] for t in tracks]


def test_the_list_says_that_several_can_be_picked(agent, offer):
    text = agent._ask_which("Pick:", offer)
    assert "several" in text and "all except" in text


@pytest.mark.parametrize("command, positions", [
    ("1 and 3", [0, 2]), ("add 1 and 3", [0, 2]), ("the first two", [0, 1]), ("2-4", [1, 2, 3]),
    ("all", [0, 1, 2, 3, 4]), ("all except the second", [0, 2, 3, 4]), ("the last two", [3, 4]),
    ("the first and the last", [0, 4]),
])
def test_several_songs_are_added(agent, say, offer, command, positions):
    text, tracks = say(agent, command)
    assert [t["track_id"] for t in tracks] == [offer[i]["track_id"] for i in positions]
    assert text.startswith("Added") and f"{len(positions)} song" in text
    assert agent._pending is None                      # the list is closed afterwards


def test_picking_by_name_adds_all_of_that_artist(agent, say, offer):
    _, tracks = say(agent, "the coldplay ones")
    assert [t["track_id"] for t in tracks] == ids(offer[:2])


def test_a_bare_artist_name_still_needs_one_match(agent, say, offer):
    text, tracks = say(agent, "Nirvana")                # two matches and no "ones": ask again
    assert "didn't catch" in text and tracks == []
    assert agent._pending is not None
    _, tracks = say(agent, "foo fighters")              # exactly one match: added
    assert ids(tracks) == [offer[4]["track_id"]]


def test_one_number_still_adds_one_song(agent, say, offer):
    text, tracks = say(agent, "2")
    assert text.startswith("Added") and ids(tracks) == [offer[1]["track_id"]]


def test_songs_already_on_the_playlist_are_skipped(agent, say, offer):
    say(agent, "add Yellow by Coldplay")
    agent._ask_which("Pick:", offer)
    text, tracks = say(agent, "1 and 2")
    assert "Already on your playlist" in text and "Yellow" in plain(text)
    assert "Added 1 song" in text
    assert len(tracks) == 2


def test_numbers_outside_the_list_are_not_guessed(agent, say, offer):
    text, tracks = say(agent, "1 and 9")
    assert "didn't catch" in text and tracks == []
    assert agent._pending is not None


def test_cancel_and_other_commands_still_work(agent, say, offer):
    text, tracks = say(agent, "cancel")
    assert "won't add" in text and tracks == []
    agent._ask_which("Pick:", offer)
    text, _ = say(agent, "show playlist")
    assert agent._pending is None


def test_asking_a_question_about_several_songs_wants_one(agent, say):
    text, _ = say(agent, "when was Iris released?")     # a list of songs called Iris
    if agent._pending is None:
        pytest.skip("Iris is not ambiguous in this catalog")
    reply, tracks = say(agent, "1 and 2")
    assert "one song" in reply and tracks == []
    assert agent._pending is not None                   # still waiting for a single pick
    reply, tracks = say(agent, "1")
    assert "released" in reply or "release" in reply
    assert tracks == []                                 # answering never adds


def test_recommend_then_take_several(agent, say):
    say(agent, "add Yellow by Coldplay")
    text, _ = say(agent, "recommend something")
    count = len(re.findall(r"^\d+\. ", plain(text), re.M))
    assert count >= 3
    reply, tracks = say(agent, "the first three")
    assert "Added 3 songs" in reply
    assert len(tracks) == 4


def test_help_mentions_taking_several(agent, say):
    text, _ = say(agent, "how do I pick a song?")
    assert "all except" in text

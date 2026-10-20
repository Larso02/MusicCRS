"""R6: recommendations based on the playlist."""

import re
import time

import pytest

from musiccrs.track_database import TrackDatabase


@pytest.fixture(autouse=True)
def _needs_r6(agent):
    if not hasattr(agent, "_recommend"):
        pytest.skip("R6 is not implemented yet")


def plain(text):
    return re.sub(r"<[^>]+>", " ", text.replace("<br>", "\n"))


def numbered(text):
    """The songs of a '1. Title - Artist (reason)' list."""
    lines = re.findall(r"^\d+\. (.+)$", plain(text), re.M)
    return [re.sub(r"\s+\((?:same|similar|popular)[^)]*\)\s*$", "", line).strip() for line in lines]


@pytest.mark.parametrize("text", [
    "recommend something", "Recommend", "recommend me some songs", "suggest songs",
    "can you recommend something", "please suggest some music", "recommend more",
    "give me some recommendations", "what should I add next", "/recommend",
    "recommend 5 songs", "suggest something for me", "more like these",
])
def test_recommend_requests_are_recognised(agent_module, text):
    assert agent_module._RECOMMEND_RE.match(text)


@pytest.mark.parametrize("text", [
    "add Recommend by Someone", "who recommended it", "what is Coldplay's most popular song",
    "recommend me some pop songs", "hello", "remove Yellow", "suggestion box",
])
def test_other_messages_are_not_recommend_requests(agent_module, text):
    assert not agent_module._RECOMMEND_RE.match(text)


def test_empty_playlist_asks_for_songs_first(agent, say):
    text, tracks = say(agent, "recommend something")
    assert "empty" in text and "add" in text.lower()
    assert tracks == []


def test_recommendations_fit_the_playlist(agent, say):
    say(agent, "add Yellow by Coldplay")
    say(agent, "add Heart-Shaped Box by Nirvana")
    text, tracks = say(agent, "recommend something")
    songs = numbered(text)
    assert 3 <= len(songs) <= 5
    assert len(set(songs)) == len(songs)
    assert not any("Yellow - Coldplay" in s or "Heart-Shaped Box - Nirvana" in s for s in songs)
    assert "similar style" in text or "same artist" in text      # every song says why
    assert len(tracks) == 2                                      # nothing added by asking


def test_variety_few_songs_from_known_artists_and_no_artist_repeated_much(agent, say):
    say(agent, "add Yellow by Coldplay")
    text, _ = say(agent, "recommend something")
    artists = [s.rsplit(" - ", 1)[-1] for s in numbered(text)]
    assert sum(a == "Coldplay" for a in artists) <= 2
    assert max(artists.count(a) for a in set(artists)) <= 2


def test_picking_a_recommendation_adds_it(agent, say):
    say(agent, "add Yellow by Coldplay")
    text, _ = say(agent, "recommend something")
    first = numbered(text)[0]
    reply, tracks = say(agent, "1")
    assert reply.startswith("Added")
    assert len(tracks) == 2
    assert f"{tracks[-1]['track_name']} - {', '.join(tracks[-1]['artist_name'])}" == first


def test_recommend_more_gives_new_songs(agent, say):
    say(agent, "add Yellow by Coldplay")
    first, _ = say(agent, "recommend something")
    second, _ = say(agent, "recommend more")
    assert not set(numbered(first)) & set(numbered(second))


def test_cancel_and_other_commands_leave_playlist_alone(agent, say):
    say(agent, "add Yellow by Coldplay")
    say(agent, "recommend something")
    reply, tracks = say(agent, "cancel")
    assert "won't add" in reply and len(tracks) == 1
    say(agent, "recommend something")
    _, tracks = say(agent, "show playlist")        # a command moves on
    assert len(tracks) == 1


def test_clearing_the_playlist_resets_recommendations(agent, say):
    say(agent, "add Yellow by Coldplay")
    first, _ = say(agent, "recommend something")
    say(agent, "clear playlist")
    say(agent, "add Yellow by Coldplay")
    again, _ = say(agent, "recommend something")
    assert numbered(first) == numbered(again)


def test_help_lists_recommendations(agent, say):
    text, _ = say(agent, "what can you do?")
    assert "Recommend" in text


def test_recommender_is_fast_and_never_returns_playlist_songs(agent_module):
    db = TrackDatabase()
    rec = agent_module.get_recommender(db)
    ids = [db.search_tracks("Yellow", "Coldplay", limit=1)[0]["track_id"],
           db.search_tracks("Heart-Shaped Box", "Nirvana", limit=1)[0]["track_id"]]
    start = time.time()
    picks = rec.recommend(ids, k=5)
    assert time.time() - start < 0.5
    assert len(picks) == 5
    assert not {i for i, _ in picks} & set(ids)
    assert rec.recommend([], k=5) == []

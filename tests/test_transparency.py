"""Part 4 transparency: the taste chips, "what do you know about me?", "forget my preferences",
and "why did you suggest X?" (answered from the stored score components)."""

import json
from types import SimpleNamespace

import pytest

from musiccrs.track_database import artist_key

router_module = pytest.importorskip("musiccrs.router")


@pytest.fixture(autouse=True)
def _needs_feature(agent):
    if not hasattr(agent, "_why"):
        pytest.skip("the transparency features are not integrated yet")


class FakeLLM:
    def __init__(self, script):
        self.script, self.calls = script, []
        self.chat = self
        self.completions = self

    def with_options(self, **options):
        return self

    def create(self, model, messages, temperature, max_tokens):
        content = messages[-1]["content"]
        self.calls.append(content)
        if content.startswith("You are MusicCRS"):
            raise TimeoutError("no phrasing")
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(self.script[content.split("MESSAGE: ")[-1]])))])


@pytest.fixture()
def ask(agent, say):
    def _ask(message, action):
        fake = FakeLLM({message: action})
        agent._llm = fake
        agent._router = router_module.Router(fake, "m")
        return say(agent, message)

    return _ask


@pytest.fixture()
def with_list(agent, say):
    say(agent, "add Yellow by Coldplay")
    say(agent, "recommend something")
    return agent


def chips(agent):
    return agent._profile_items()


# ---------- the data behind the chips ----------
def test_profile_items_are_empty_for_a_new_chat(agent):
    assert chips(agent) == []


def test_profile_items_show_what_was_liked_and_rejected(with_list):
    agent = with_list
    agent._profile.like_tag("grunge")
    agent._profile.dislike_artist(artist_key("Coldplay"))
    agent._profile.like_track(agent._last_recs[0]["track_id"])
    agent._profile.add_wish("more upbeat")
    slots = chips(agent)
    assert ("like", "grunge") in slots and ("like", "more upbeat") in slots
    assert any(slot == "dislike" and "oldplay" in text for slot, text in slots)
    assert any(slot == "like" and " by " in text for slot, text in slots)        # a liked song, with its artist


def test_every_reply_carries_the_profile_for_the_ui(agent):
    agent._profile.like_tag("indie rock")
    agent._dialogue_connector.sent.clear()
    agent.receive_utterance(SimpleNamespace(text="show playlist"))
    acts = [a for a in agent._dialogue_connector.sent[-1].dialogue_acts if str(a.intent) == "PROFILE_STATE"]
    assert len(acts) == 1
    assert [(a.slot, a.value) for a in acts[0].annotations] == [("like", "indie rock")]


def test_the_chips_are_capped(agent):
    for i in range(40):
        agent._profile.like_tag(f"tag {i}")
    assert len(chips(agent)) == 16


# ---------- what do you know about me / forget ----------
def test_nothing_learned_yet(agent, say):
    text, _ = say(agent, "what do you know about my taste?")
    assert "haven't learned anything" in text


def test_preferences_are_listed_and_can_be_forgotten(agent, say):
    agent._profile.like_tag("grunge")
    agent._profile.dislike_tag("rap")
    text, _ = say(agent, "/preferences")
    assert "you like grunge" in text and "you don't want rap" in text
    say(agent, "add Clocks by Coldplay")
    text, tracks = say(agent, "/forget")
    assert "forgotten" in text and agent._profile.is_empty()
    assert len(tracks) == 1                                       # the playlist is untouched


@pytest.mark.parametrize("text", ["what do you know about me?", "what have you learned about my taste", "show my preferences"])
def test_preference_phrasings_without_the_llm(agent, say, text):
    assert "haven't learned" in say(agent, text)[0]


@pytest.mark.parametrize("text", ["forget my preferences", "reset my taste", "please forget all preferences"])
def test_forget_phrasings_without_the_llm(agent, say, text):
    agent._profile.like_tag("rock")
    assert "forgotten" in say(agent, text)[0] and agent._profile.is_empty()


def test_the_llm_actions(ask, agent):
    agent._profile.like_tag("grunge")
    text, _ = ask("so what have you figured out about me", {"action": "preferences"})
    assert "grunge" in text
    text, _ = ask("let's wipe what you learned", {"action": "forget"})
    assert agent._profile.is_empty() and "forgotten" in text


def test_asking_what_was_learned_keeps_the_open_list(ask, with_list):
    agent = with_list
    assert agent._pending
    ask("what do you know about me", {"action": "preferences"})
    assert agent._pending                                          # the user can still pick a number


# ---------- why ----------
def test_why_a_song_from_the_list(with_list, say):
    agent = with_list
    first = agent._last_recs[0]
    reason = agent._reasons[first["track_id"]]
    text, _ = say(agent, f"why did you suggest {first['track_name']}")
    assert first["track_name"] in text and reason.split(";")[0] in text and "Reason:" in text


def test_why_the_whole_list(with_list, say):
    agent = with_list
    text, _ = say(agent, "why did you recommend these")
    assert text.count("<br>") >= len(agent._last_recs) and "Why I suggested these" in text


def test_why_before_anything_was_suggested(agent, say):
    text, _ = say(agent, "why did you recommend these?")
    assert "haven't suggested anything" in text


def test_why_a_song_that_was_not_suggested(with_list, say):
    agent = with_list
    text, _ = say(agent, "why did you suggest Bohemian Rhapsody")
    assert "didn't suggest" in text


def test_why_a_song_the_user_added_himself(agent, say):
    say(agent, "add Clocks by Coldplay")
    text, _ = say(agent, "why did you suggest Clocks")
    assert "on your playlist, but I didn't suggest it" in text


def test_why_through_the_llm_resolves_the_position(ask, with_list):
    agent = with_list
    second = agent._last_recs[1]
    text, _ = ask("why is the second one here?", {"action": "why", "title": second["track_name"], "artist": second["artist_name"][0]})
    assert second["track_name"] in text and "Reason:" in text


def test_reasons_follow_the_refined_recommendations(with_list, ask):
    agent = with_list
    ask("no more coldplay, more grunge", {"action": "feedback", "disliked_artists": ["Coldplay"], "liked_tags": ["grunge"]})
    for track in agent._last_recs:
        assert track["track_id"] in agent._reasons and agent._reasons[track["track_id"]]


def test_clearing_forgets_the_reasons(with_list, say):
    agent = with_list
    say(agent, "/clear")
    assert agent._reasons == {} and agent._last_recs == []

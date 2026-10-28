"""LLM-first understanding: a scripted fake LLM stands in for the real model."""

import json
from types import SimpleNamespace

import pytest

from musiccrs.track_database import TrackDatabase

router = pytest.importorskip("musiccrs.router")


class FakeLLM:
    """Answers each message with a scripted reply (a dict, a raw string or an exception)."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.options = None
        self.chat = self
        self.completions = self

    def with_options(self, **options):
        self.options = options
        return self

    def create(self, model, messages, temperature, max_tokens):
        user = messages[-1]["content"]
        self.calls.append(user)
        reply = self.script[user.split("MESSAGE: ")[-1]]
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class StubGenerator:
    def __init__(self, track_ids):
        self.track_ids = track_ids
        self.requests = []

    def generate(self, description):
        self.requests.append(description)
        return {"track_ids": self.track_ids, "actual_minutes": 60.0}


@pytest.fixture(autouse=True)
def _needs_router(agent):
    if not hasattr(agent, "_route"):
        pytest.skip("the router is not integrated yet")


@pytest.fixture()
def ask(agent, say):
    """ask(agent_message, scripted_action) -> (reply, playlist) with the fake LLM in place."""

    def _ask(message, action):
        llm = agent._router.llm if agent._router else None
        script = dict(getattr(llm, "script", {}))
        script[message] = action
        fake = FakeLLM(script)
        agent._llm = fake
        agent._router = router.Router(fake, "test-model")
        return say(agent, message)

    return _ask


# ---------- the parts of the router ----------
@pytest.mark.parametrize("text, action", [
    ('{"action": "view"}', "view"),
    ('```json\n{"action": "add", "title": "Yellow"}\n```', "add"),
    ('Sure! {"action": "clear"} Hope that helps', "clear"),
])
def test_parse_action_reads_json_in_a_reply(text, action):
    assert router.parse_action(text)["action"] == action


@pytest.mark.parametrize("text", ["", "no json here", '{"action": "dance"}', '{"title": "Yellow"}', "[1, 2]", '{"action": '])
def test_parse_action_rejects_unusable_replies(text):
    assert router.parse_action(text) is None


def test_the_state_is_part_of_the_prompt():
    prompt = router.build_user_message("the first two", {
        "playlist": ["Yellow - Coldplay"], "last": "Yellow - Coldplay",
        "open_list": ["Sparks - Coldplay", "Trouble - Coldplay"], "pending_confirm": None})
    assert "PLAYLIST: 1. Yellow - Coldplay" in prompt and "LAST SONG: Yellow - Coldplay" in prompt
    assert "1. Sparks - Coldplay | 2. Trouble - Coldplay" in prompt and prompt.endswith("MESSAGE: the first two")
    assert "PENDING QUESTION" not in prompt
    assert "(empty)" in router.build_user_message("hi", {})


def test_router_returns_none_when_the_llm_fails():
    fake = FakeLLM({"hi": TimeoutError("slow")})
    assert router.Router(fake, "m").route("hi", {}) is None


def test_router_allows_one_short_attempt():
    fake = FakeLLM({"hi": {"action": "chat", "reply": "Hello"}})
    assert router.Router(fake, "m", timeout=2.5).route("hi", {})["reply"] == "Hello"
    assert fake.options == {"timeout": 2.5, "max_retries": 0}


# ---------- every action ----------
def test_add(ask, agent):
    text, tracks = ask("could you throw clocks by coldplay on there",
                       {"action": "add", "title": "Clocks", "artist": "Coldplay"})
    assert text.startswith("Added") and tracks[0]["track_name"] == "Clocks"


def test_add_fixes_a_typo_through_the_database(ask):
    text, tracks = ask("stnad by me please", {"action": "add", "title": "stnad by me", "artist": None})
    assert tracks and tracks[0]["track_name"] == "Stand By Me"


def test_add_an_ambiguous_title_opens_the_choice_list(ask, agent):
    text, tracks = ask("put Love on it", {"action": "add", "title": "Love", "artist": None})
    assert "Several songs" in text and tracks == [] and agent._pending


def test_view_and_clear(ask, say, agent):
    say(agent, "add Yellow by Coldplay")
    text, _ = ask("what have I got?", {"action": "view"})
    assert "Yellow" in text
    text, tracks = ask("nuke it all", {"action": "clear"})
    assert "cleared" in text and tracks == []


def test_remove_by_artist(ask, say, agent):
    say(agent, "add Yellow by Coldplay")
    say(agent, "add Smells Like Teen Spirit by Nirvana")
    text, tracks = ask("get rid of the nirvana one", {"action": "remove", "title": None, "artist": "Nirvana"})
    assert text.startswith("Removed") and [t["track_name"] for t in tracks] == ["Yellow"]


def test_help(ask):
    text, _ = ask("what can you actually do?", {"action": "help", "topic": None})
    assert "Add songs" in text
    text, _ = ask("how do I take a song off?", {"action": "help", "topic": "remove"})
    assert "Remove" in text


@pytest.mark.parametrize("message, action, expected", [
    ("who made it?", {"action": "ask", "kind": "artist", "subject": "it", "subject_type": "last"}, "Coldplay"),
    ("how long is Clocks by Coldplay", {"action": "ask", "kind": "length", "subject": "Clocks by Coldplay", "subject_type": "track"}, "long"),
    ("when did Yellow by Coldplay come out", {"action": "ask", "kind": "year", "subject": "Yellow by Coldplay", "subject_type": "track"}, "released"),
    ("how many songs has Coldplay got", {"action": "ask", "kind": "count", "subject": "Coldplay", "subject_type": "artist"}, "songs by"),
    ("what is Coldplay's biggest hit", {"action": "ask", "kind": "top", "subject": "Coldplay", "subject_type": "artist"}, "most popular"),
    ("tell me about Nirvana", {"action": "ask", "kind": "about", "subject": "Nirvana", "subject_type": "artist"}, "Nirvana"),
])
def test_questions(ask, say, agent, message, action, expected):
    say(agent, "add Yellow by Coldplay")                   # "it" = the last song
    text, _ = ask(message, action)
    assert expected in text


def test_a_mixed_up_question_kind_is_still_understood(ask):
    text, _ = ask("how many tracks by Coldplay?", {"action": "ask", "kind": "count", "subject": "Coldplay", "subject_type": "track"})
    assert "songs by" in text


def test_recommend_then_select_from_the_open_list(ask, say, agent):
    say(agent, "add Yellow by Coldplay")
    text, _ = ask("got anything similar?", {"action": "recommend", "hint": None})
    assert "you might like" in text and agent._pending
    text, tracks = ask("the first and the third", {"action": "select", "positions": [1, 3]})
    assert text.startswith("Added 2 songs") and len(tracks) == 3
    assert agent._pending is None


def test_select_everything_except(ask, say, agent):
    say(agent, "add Yellow by Coldplay")
    ask("more please", {"action": "recommend", "hint": None})
    text, tracks = ask("all but the last one", {"action": "select", "mode": "all", "except": [5]})
    assert text.startswith("Added 4 songs") and len(tracks) == 5


def test_select_none_closes_the_list(ask, say, agent):
    say(agent, "add Yellow by Coldplay")
    ask("more please", {"action": "recommend", "hint": None})
    text, tracks = ask("nah skip those", {"action": "select", "mode": "none"})
    assert "won't add" in text and len(tracks) == 1 and agent._pending is None


def test_a_number_outside_the_list_is_not_guessed(ask, say, agent):
    say(agent, "add Yellow by Coldplay")
    ask("more please", {"action": "recommend", "hint": None})
    text, tracks = ask("1 and 9", {"action": "select", "positions": [1, 9]})
    assert "didn't catch" in text and len(tracks) == 1 and agent._pending


def test_select_without_an_open_list_falls_back_to_the_rules(ask):
    text, _ = ask("the first two", {"action": "select", "positions": [1, 2]})
    assert "don't understand" in text


def test_another_request_closes_an_open_list(ask, say, agent):
    say(agent, "add Yellow by Coldplay")
    ask("more please", {"action": "recommend", "hint": None})
    ask("show me what I have", {"action": "view"})
    assert agent._pending is None


def test_generate_asks_before_replacing_then_confirm(ask, say, agent):
    db = TrackDatabase()
    ids = [db.search_tracks("Clocks", "Coldplay", limit=1)[0]["track_id"]]
    agent._playlist_generator = StubGenerator(ids)
    say(agent, "add Yellow by Coldplay")
    text, tracks = ask("I want something for the gym", {"action": "generate", "description": "the gym"})
    assert "replaces your current playlist" in text and len(tracks) == 1
    text, tracks = ask("yeah go for it", {"action": "confirm", "answer": "yes"})
    assert "Created a new playlist" in text and [t["track_name"] for t in tracks] == ["Clocks"]


def test_confirm_no_keeps_the_playlist(ask, say, agent):
    agent._playlist_generator = StubGenerator([])
    say(agent, "add Yellow by Coldplay")
    ask("a playlist for studying", {"action": "generate", "description": "studying"})
    text, tracks = ask("no keep it", {"action": "confirm", "answer": "no"})
    assert "keep your current playlist" in text and len(tracks) == 1


def test_confirm_without_a_question_falls_back(ask):
    text, _ = ask("yes", {"action": "confirm", "answer": "yes"})
    assert "don't understand" in text


def test_another_request_cancels_the_replace_question(ask, say, agent):
    agent._playlist_generator = StubGenerator([])
    say(agent, "add Yellow by Coldplay")
    ask("a playlist for studying", {"action": "generate", "description": "studying"})
    ask("show me my songs", {"action": "view"})
    assert agent._confirm_generate is None


def test_feedback_is_recorded_and_answered_with_new_recommendations(ask, say, agent):
    say(agent, "add Yellow by Coldplay")
    text, _ = ask("less metal, more indie", {"action": "feedback", "wish": "less metal, more indie"})
    assert agent._feedback and agent._feedback[0]["wish"] == "less metal, more indie"
    assert "you might like" in text


def test_chat_reply_is_escaped(ask):
    text, _ = ask("hello there", {"action": "chat", "reply": "Hi <b>there</b> & welcome"})
    assert text == "Hi &lt;b&gt;there&lt;/b&gt; &amp; welcome"


def test_quit(ask, agent):
    ask("ok I'm done, bye", {"action": "quit"})
    assert "Bye" in agent._dialogue_connector.sent[-1].text


# ---------- the fallback ----------
def test_slash_commands_never_use_the_llm(ask, say, agent):
    ask("hello", {"action": "chat", "reply": "Hi"})                      # installs the fake
    llm = agent._router.llm
    text, tracks = say(agent, "/clear")
    assert "cleared" in text and llm.calls == [llm.calls[0]]


def test_button_clicks_never_use_the_llm(ask, say, agent):
    say(agent, "add Yellow by Coldplay")
    ask("more please", {"action": "recommend", "hint": None})
    llm = agent._router.llm
    calls = len(llm.calls)
    label = agent._plain_label(agent._pending[0])
    text, tracks = say(agent, label)
    assert text.startswith("Added") and len(llm.calls) == calls


@pytest.mark.parametrize("bad", [TimeoutError("slow"), "I cannot help with that", '{"action": "dance"}', "{broken"])
def test_the_rules_answer_when_the_llm_fails(ask, agent, bad):
    text, tracks = ask("add Clocks by Coldplay", bad)
    assert text.startswith("Added") and tracks[0]["track_name"] == "Clocks"


def test_an_action_without_its_fields_falls_back(ask):
    text, tracks = ask("add Clocks by Coldplay", {"action": "add", "title": None, "artist": None})
    assert text.startswith("Added")


def test_without_an_llm_nothing_changes(agent, say):
    assert agent._router is None
    text, tracks = say(agent, "add Clocks by Coldplay")
    assert text.startswith("Added")


def test_the_prompt_receives_the_conversation_state(ask, say, agent):
    say(agent, "add Yellow by Coldplay")
    say(agent, "add Clocks by Coldplay")
    ask("more please", {"action": "recommend", "hint": None})
    ask("hello", {"action": "chat", "reply": "Hi"})
    prompt = agent._router.llm.calls[-1]
    assert "PLAYLIST: 1. Yellow by Coldplay | 2. Clocks by Coldplay" in prompt
    assert "LAST SONG: Clocks by Coldplay" in prompt
    assert "OPEN LIST: 1. " in prompt                      # the recommendations are still open

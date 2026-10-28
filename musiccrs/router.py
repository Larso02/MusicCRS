"""LLM-first understanding of what the user wants.

The LLM reads the message together with a short summary of the conversation state and answers
with one JSON action. The agent carries the action out with its own code and database, so the
LLM never supplies songs or facts. If the LLM is missing, slow or unclear, route() returns None
and the agent falls back to its rule-based parsers.

    Router(llm).route("could you throw stnad by me on there", state)
    -> {"action": "add", "title": "Stand By Me", "artist": None}
"""

import json
import logging
import re

log = logging.getLogger(__name__)

ACTIONS = {"add", "remove", "view", "clear", "help", "ask", "recommend", "select",
           "generate", "confirm", "feedback", "chat", "quit"}

TIMEOUT_SECONDS = 4.0      # beyond this the rule-based parsers answer instead

SYSTEM = """You are the language understanding part of MusicCRS, a chat system that manages a music playlist.
Read the user's message and the state, then answer with ONE JSON object and nothing else.

Actions and their fields:
- add: title, artist (null if not said)
- remove: title (null if not said), artist (null if not said)
- view: show the playlist
- clear: empty the playlist
- help: topic (null, or a word such as "add", "remove", "recommend")
- ask: kind, subject, subject_type. subject_type is "track", "artist" or "last" (the song just discussed: "it", "this song"). For a song, write the subject as "Title by Artist" whenever the user gave the artist or the song is on the PLAYLIST. kind for tracks: artist, album, year, length, popularity, tags, about. kind for artists: songs, albums, count, top, tags, about.
- recommend: hint (an extra wish such as "more mellow", or null)
- select: choose from the OPEN LIST. Use positions (list of numbers), or mode "all" / "none", and "except" (list of numbers) to leave some out. Only when an open list exists.
- generate: description (a whole new playlist from a description, for example "gym" or "sad love songs")
- confirm: answer "yes" or "no" (only when a question is pending)
- feedback: liked (list of strings), disliked (list of strings), wish (text): opinions about recommended songs or styles
- quit: the user wants to end the conversation
- chat: anything else (greetings, thanks, off-topic); put a short friendly reply in "reply"

Rules:
- Copy names the user wrote and fix obvious typos.
- "it", "this", "that song" mean the last song.
- When the user points at a song of the PLAYLIST ("the second one", "the nirvana song"), write that song's real title and artist in the fields, never its number.
- If an OPEN LIST exists and the message picks from it, use select.
- Answer with JSON only: {"action": "...", ...fields}"""


def build_user_message(message: str, state: dict) -> str:
    """The state the LLM needs to resolve references like 'the first two' or 'yes'."""
    playlist = " | ".join(f"{i}. {t}" for i, t in enumerate(state.get("playlist") or [], 1))
    lines = ["PLAYLIST: " + (playlist or "(empty)"),
             "LAST SONG: " + (state.get("last") or "(none)")]
    if state.get("open_list"):
        lines.append("OPEN LIST: " + " | ".join(f"{i}. {t}" for i, t in enumerate(state["open_list"], 1)))
    if state.get("pending_confirm"):
        lines.append("PENDING QUESTION: " + state["pending_confirm"])
    lines.append("MESSAGE: " + message)
    return "\n".join(lines)


def parse_action(text: str) -> dict | None:
    """The JSON action in the LLM's reply, or None if it is missing or not a known action."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        action = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(action, dict) or action.get("action") not in ACTIONS:
        return None
    return action


class Router:
    def __init__(self, llm, model: str, timeout: float = TIMEOUT_SECONDS):
        self.model = model
        # one attempt only: a retry would use up the time budget
        self.llm = llm.with_options(timeout=timeout, max_retries=0) if hasattr(llm, "with_options") else llm

    def route(self, message: str, state: dict) -> dict | None:
        try:
            reply = self.llm.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_user_message(message, state)},
                ],
            )
            return parse_action(reply.choices[0].message.content)
        except Exception:                       # timeout, network error, odd reply
            log.warning("Router failed; using the rule-based parsers.", exc_info=True)
            return None

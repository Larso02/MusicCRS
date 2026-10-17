
"""MusicCRS conversational agent."""
import html
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.platforms import FlaskSocketPlatform

from track_database import TrackDatabase, norm


load_dotenv()

LLM_BASE_URL = "https://openwebui.ux.uis.no/api"
LLM_MODEL = "gorina10.gemma-4-31b"
LLM_API_KEY = os.getenv("MY_OLLAMA_API_KEY")

_INTENT_OPTIONS = Intent("OPTIONS")
_INTENT_PLAYLIST = Intent("PLAYLIST_STATE")
_ADD_RE = re.compile(
    r"^(?:/add\s+|(?:please\s+)?(?:add|put|queue)\s+(?:the\s+(?:song|track|)\s+)?)"
    r"(?P<q>.+?)(?:\s+(?:to|on|in|into)\s+(?:my|the)\s+(?:play ?list|queue))?\s*$",
    re.I
)
_REMOVE_RE = re.compile(
    r"^(?:/remove\s+|/del\s+|(?:please\s+)?(?:remove|delete|drop|take out)\s+"
    r"(?:the\s+(?:song|track)\s+)?)"
    r"(?P<q>.+?)(?:\s+from\s+(?:my|the)\s+(?:play ?list|queue))?\s*$",
    re.I,
)

_VIEW = ("/playlist", "/list", "show playlist", "show my playlist", "view playlist", "what's in my playlist", "what is in my playlist")
_CLEAR = ("/clear", "clear playlist", "clear my playlist", "empty playlist")
class MusicCRS(Agent):
    def __init__(self, use_llm: bool = True):
        """Initialize MusicCRS agent."""
        super().__init__(id="MusicCRS")

        if use_llm:
            self._llm = OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
            )
        else:
            self._llm = None

        # TalkPlay database containing verified tracks.
        self._db = TrackDatabase()

        # Store only track IDs, never arbitrary song names.
        self._playlist: list[str] = []

    def welcome(self) -> None:
        """Send the agent's welcome message."""
        utterance = AnnotatedUtterance(
            "Hello, I'm MusicCRS. What are you in the mood for?",
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def goodbye(self) -> None:
        """End the conversation."""
        utterance = AnnotatedUtterance(
            "It was nice talking to you. Bye",
            dialogue_acts=[DialogueAct(intent=self.stop_intent)],
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    # --------------------------------------------------
    # Playlist helpers
    # --------------------------------------------------

    def _playlist_tracks(self) -> list[dict]:
        return self._db.get_tracks(self._playlist)

    @staticmethod
    def _track_label(track: dict) -> str:
        """Return a readable track title and artist."""
        artists = ", ".join(track["artist_name"])
        return html.escape(f"{track['track_name']} - {artists}")

    def _playlist_act(self) -> DialogueAct:
        return DialogueAct(
            intent=_INTENT_PLAYLIST,
            annotations=[
                SlotValueAnnotation("track", json.dumps(t, ensure_ascii=False)) for t in self._playlist_tracks()
            ],
        )

    # --------------------------------------------------
    # Add track
    # --------------------------------------------------

    def _add_track(self, query: str) -> str:
        """Add a verified TalkPlay track to the playlist."""
        query = query.strip()

        if not query:
            return "Please provide a track name."
            

        # First check whether the input is a track ID.
        track = self._db.get_track(query)
        others = 0

        if track is None:
            matches = self._db.find_tracks(query, limit=10)
            if not matches:
                return f"I couldn't find '{html.escape(query)}' in the catalog."
            
            exact = [m for m in matches if m["exact"]]
            if not exact:
                suggestions = ", ".join(
                    self._track_label(m)
                    for m in matches[:3]
                )

                return (
                    f"No exact match for '{html.escape(query)}'. "
                    f"Did you mean: {suggestions}?"
                )

            track = exact[0]
            others = len(exact) - 1

        
        track_id = track["track_id"]
        if track_id in self._playlist:
            return (
                f"{self._track_label(track)} "
                "is already in your playlist."
            )

        self._playlist.append(track_id)
        message = (
            f"Added {self._track_label(track)} to your playlist."
        )
        if others:
            noun = "track shares" if others == 1 else "tracks share"
            message += f" {others} other {noun} this title; say 'add <title> by <artist>' to pick another."


        return message

    # --------------------------------------------------
    # Remove track
    # --------------------------------------------------

    def _remove_track(self, query: str) -> str:
        """Remove a playlist track by ID or title and artist."""
        query = query.strip()

        if not query:
            return (
                "Please specify a track to remove."
            )

        # React Remove buttons send the unique track ID.
        if query in self._playlist:
            track = self._db.get_track(query)
            self._playlist.remove(query)

            return f"Removed {self._track_label(track)}." if track else "Removed track."

        # Support title-based removal:
        # "Yellow"
        # "Yellow by Coldplay"
        words = norm(re.sub(r"\s+by\s+", " ", query, flags=re.I)).split()
        if not words:
            return "Please specify a track to remove."

        matches = [
            t
            for t in self._playlist_tracks()
            if set(words) <= set(norm(t["track_name"] + " " + " ".join(t["artist_name"])).split())
        ]

        if not matches:
            return f"Could not find '{html.escape(query)}' in your playlist."
            

        if len(matches) > 1:
            choices = ", ".join(
                self._track_label(t)
                for t in matches
            )

            return (
                f"Multiple tracks match: {choices}. "
                "Please specify the artist or use the Remove button."
            )

        track = matches[0]
        self._playlist.remove(track["track_id"])

        return (
            f"Removed {self._track_label(track)}."
        )

    # --------------------------------------------------
    # Dialogue handling
    # --------------------------------------------------

    def receive_utterance(self, utterance: Utterance) -> None:
        """Handle a new user message."""
        dialogue_acts = []
        command = utterance.text.strip()
        lowered = command.lower()
        m_add, m_remove = _ADD_RE.match(command), _REMOVE_RE.match(command)

        # View playlist.
        if lowered in _VIEW:
            tracks = self._playlist_tracks()
            if not tracks:
                response = "Your playlist is empty."
            else:
                listing = "<br>".join(
                    f"{i}. {self._track_label(t)}" for i, t in enumerate(tracks, start=1)
                )
                word = "song" if len(tracks) == 1 else "songs"
                response = f"Your playlist contains {len(tracks)} {word}:<br>{listing}"
            

        # Clear playlist.
        elif lowered in _CLEAR:
            self._playlist.clear()
            response = "Your playlist has been cleared."

        # Add a verified track.
        elif m_add:
            response = self._add_track(m_add.group("q"))

        # Remove a track.
        elif m_remove:
            response = self._remove_track(m_remove.group("q"))

        # Information about the agent.
        elif lowered == "/info":
            response = self._info()

        # Direct LLM interaction.
        elif lowered.startswith("/ask_llm "):
            prompt = command[len("/ask_llm "):]
            response = self._ask_llm(prompt)

        # DialogueKit quick reply options.
        elif lowered == "/options":
            options = [
                "Play some jazz music",
                "Recommend me some pop songs",
                "Create a workout playlist",
            ]

            response = self._options(options)

            dialogue_acts = [
                DialogueAct(
                    intent=_INTENT_OPTIONS,
                    annotations=[
                        SlotValueAnnotation("option", option)
                        for option in options
                    ],
                )
            ]

        # End conversation.
        elif lowered == "/quit":
            self.goodbye()
            return

        else:
            response = (
                "I'm sorry, I don't understand that command."
            )
        dialogue_acts.append(self._playlist_act())
        self._dialogue_connector.register_agent_utterance(
            AnnotatedUtterance(
                response,
                participant=DialogueParticipant.AGENT,
                dialogue_acts=dialogue_acts,
            )
        )

    # --------------------------------------------------
    # Other response handlers
    # --------------------------------------------------

    def _info(self) -> str:
        """Describe the MusicCRS agent."""
        return (
            "I am MusicCRS, a conversational recommender "
            "system for music."
        )

    def _ask_llm(self, prompt: str) -> str:
        """Send a prompt to the configured LLM."""
        if not self._llm:
            return "The agent is not configured to use an LLM."

        llm_response = self._llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.7,
        )

        return (
            f"LLM response: "
            f"{llm_response.choices[0].message.content}"
        )

    def _options(self, options: list[str]) -> str:
        """Present DialogueKit quick reply options."""
        return (
            "Here are some options:\n<ol>\n"
            + "\n".join(
                f"<li>{option}</li>"
                for option in options
            )
            + "</ol>\n"
        )


if __name__ == "__main__":
    platform = FlaskSocketPlatform(MusicCRS)
    platform.start()

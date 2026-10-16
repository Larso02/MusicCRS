
"""MusicCRS conversational agent."""

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

from track_database import TrackDatabase


load_dotenv()

LLM_BASE_URL = "https://openwebui.ux.uis.no/api"
LLM_MODEL = "gorina10.gemma-4-31b"
LLM_API_KEY = os.getenv("MY_OLLAMA_API_KEY")

_INTENT_OPTIONS = Intent("OPTIONS")


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
        """Resolve playlist IDs into database track records."""
        tracks = []

        for track_id in self._playlist:
            track = self._db.get_track(track_id)

            if track is not None:
                tracks.append(track)

        return tracks

    @staticmethod
    def _track_label(track: dict) -> str:
        """Return a readable track title and artist."""
        artists = ", ".join(track["artist_name"])
        return f"{track['track_name']} — {artists}"

    def _playlist_response(self, message: str) -> str:
        """Send the current playlist state to the React frontend."""
        return "__PLAYLIST_STATE__" + json.dumps(
            {
                "message": message,
                "tracks": self._playlist_tracks(),
            },
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # Add track
    # --------------------------------------------------

    def _add_track(self, query: str) -> str:
        """Add a verified TalkPlay track to the playlist."""
        query = query.strip()

        if not query:
            return self._playlist_response(
                "Please provide a track name."
            )

        # First check whether the input is a track ID.
        track = self._db.get_track(query)

        if track is None:
            # Support:
            # "Yellow"
            # "Yellow by Coldplay"
            parts = re.split(
                r"\s+by\s+",
                query,
                maxsplit=1,
                flags=re.IGNORECASE,
            )

            title = parts[0].strip()
            artist = parts[1].strip() if len(parts) == 2 else None

            matches = self._db.search_tracks(
                title,
                artist=artist,
                limit=10,
            )

            if not matches:
                return self._playlist_response(
                    f"I couldn't find '{query}' in the TalkPlay catalog."
                )

            # Prefer an exact title match.
            exact_matches = [
                candidate
                for candidate in matches
                if candidate["track_name"].casefold()
                == title.casefold()
            ]

            if not exact_matches:
                suggestions = ", ".join(
                    self._track_label(candidate)
                    for candidate in matches[:3]
                )

                return self._playlist_response(
                    f"No exact match for '{query}'. "
                    f"Possible matches: {suggestions}"
                )

            # Search results are ranked by popularity.
            # R4 will add explicit ambiguity handling.
            track = exact_matches[0]

        track_id = track["track_id"]

        if track_id in self._playlist:
            return self._playlist_response(
                f"{self._track_label(track)} "
                "is already in your playlist."
            )

        self._playlist.append(track_id)

        return self._playlist_response(
            f"Added {self._track_label(track)}."
        )

    # --------------------------------------------------
    # Remove track
    # --------------------------------------------------

    def _remove_track(self, query: str) -> str:
        """Remove a playlist track by ID or title and artist."""
        query = query.strip()

        if not query:
            return self._playlist_response(
                "Please specify a track to remove."
            )

        # React Remove buttons send the unique track ID.
        if query in self._playlist:
            track = self._db.get_track(query)
            self._playlist.remove(query)

            if track is not None:
                return self._playlist_response(
                    f"Removed {self._track_label(track)}."
                )

            return self._playlist_response("Track removed.")

        # Support title-based removal:
        # "Yellow"
        # "Yellow by Coldplay"
        parts = re.split(
            r"\s+by\s+",
            query,
            maxsplit=1,
            flags=re.IGNORECASE,
        )

        title = parts[0].strip().casefold()
        artist = (
            parts[1].strip().casefold()
            if len(parts) == 2
            else None
        )

        matches = [
            track
            for track in self._playlist_tracks()
            if track["track_name"].casefold() == title
            and (
                artist is None
                or any(
                    artist in name.casefold()
                    for name in track["artist_name"]
                )
            )
        ]

        if not matches:
            return self._playlist_response(
                f"Could not find '{query}' in your playlist."
            )

        if len(matches) > 1:
            choices = ", ".join(
                self._track_label(track)
                for track in matches
            )

            return self._playlist_response(
                f"Multiple tracks match '{query}': {choices}. "
                "Please specify the artist or use the Remove button."
            )

        track = matches[0]
        self._playlist.remove(track["track_id"])

        return self._playlist_response(
            f"Removed {self._track_label(track)}."
        )

    # --------------------------------------------------
    # Dialogue handling
    # --------------------------------------------------

    def receive_utterance(self, utterance: Utterance) -> None:
        """Handle a new user message."""
        response = ""
        dialogue_acts = []

        command = utterance.text.strip()
        lowered = command.lower()

        # View playlist.
        if lowered in (
            "/playlist",
            "show playlist",
            "show my playlist",
            "view playlist",
            "what is in my playlist",
            "what's in my playlist",
        ):
            tracks = self._playlist_tracks()

            if not tracks:
                response = self._playlist_response(
                    "Your playlist is empty."
                )
            else:
                listing = "\n".join(
                    f"{i}. {self._track_label(track)}"
                    for i, track in enumerate(tracks, start=1)
                )

                count = len(tracks)
                word = "song" if count == 1 else "songs"

                response = self._playlist_response(
                    f"Your playlist contains "
                    f"{count} {word}:\n{listing}"
                )

        # Clear playlist.
        elif lowered in (
            "/clear",
            "clear playlist",
            "clear my playlist",
            "empty playlist",
        ):
            self._playlist.clear()

            response = self._playlist_response(
                "Playlist cleared."
            )

        # Add a verified track.
        elif lowered.startswith("/add ") or lowered.startswith("add "):
            query = re.sub(
                r"^/?add\s+",
                "",
                command,
                flags=re.IGNORECASE,
            ).strip()

            response = self._add_track(query)

        # Remove a track.
        elif lowered.startswith("/remove ") or lowered.startswith("remove "):
            query = re.sub(
                r"^/?remove\s+",
                "",
                command,
                flags=re.IGNORECASE,
            ).strip()

            response = self._remove_track(query)

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

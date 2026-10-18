
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
from fuzzy import FuzzyTrackIndex
from capabilities import help_answer, is_help_request

load_dotenv()

LLM_BASE_URL = "https://openwebui.ux.uis.no/api"
LLM_MODEL = "gorina10.gemma-4-31b"
LLM_API_KEY = os.getenv("MY_OLLAMA_API_KEY")

FUZZY_ACCEPT = 0.9
FUZZY_SUGGEST = 0.6
_FUZZY_INDEX = None

def get_fuzzy_index(db):
    global _FUZZY_INDEX
    if _FUZZY_INDEX is None:
        _FUZZY_INDEX = FuzzyTrackIndex(db)
    return _FUZZY_INDEX

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

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
_CANCEL = {"cancel", "no", "none", "neither", "nevermind", "never mind", "skip", "no thanks", "no thank you", "nope", "not now", "not this time", "stop", "quit", "exit", "leave", "dismiss", "forget it", "forget about it", "forget that", "forget this", "forget it all", "forget everything", "forget all of this", "forget everything I said", "forget everything I just said", "forget everything I just told you", "forget everything I just told you about my playlist", "forget everything I just told you about my music preferences"}
_FILLER = {"the", "one", "ones", "please", "i", "want", "pick", "choose", "select", "take", "song", "track", "by", "a", "with", "number", "option", "it", "that", "go", "for", "me", "my", "to", "in", "on", "of", "and", "or", "but", "so", "if", "then", "else", "when", "where", "how", "what", "which", "who", "whom", "whose", "why", "because", "since", "although", "though", "even though"}


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
        self._pending: list[dict] | None = None
        self._offer: list[dict] | None = None

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

    def _fuzzy_guesses(self, query: str) -> list[tuple[dict, float]]:
        found = get_fuzzy_index(self._db).search(query, k=5)
        tracks = {t["track_id"]: t for t in self._db.get_tracks([i for i, _ in found])}
        return [(tracks[i], sim) for i, sim in found if i in tracks]

    def _add_chosen(self, track:dict, note: str="") -> str:
        if track["track_id"] in self._playlist:
            return f"{self._track_label(track)} is already in your playlist."
        self._playlist.append(track["track_id"])
        return f"Added {self._track_label(track)} to your playlist.{note}"

    @staticmethod
    def _plain_label(track: dict) -> str:
        return f"{track['track_name']} by {', '.join(track['artist_name'])}"


    @staticmethod
    def _distinct_songs(tracks: list[dict]) -> list[dict]:
        seen, out = set(), []
        for t in tracks:
            key = (norm(t["track_name"]), norm(" ".join(t["artist_name"])))
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def _rank_for_playlist(self, tracks: list[dict]) -> list[dict]:
        on_list = self._playlist_tracks()
        artists = {a for t in on_list for a in t["artist_name"]}
        albums = {a for t in on_list for a in t["album_name"]}

        def score(t):
            return (
                t["popularity"] / 100
                + 1.0 * bool(artists & set(t["artist_name"]))
                + 0.5 * bool(albums & set(t["album_name"]))
            )

        return sorted(tracks, key=score, reverse=True)

    def _ask_which(self, propmt: str, tracks: list[dict]) -> str:
        self._pending = tracks
        self._offer = tracks
        listing = "<br>".join(
            f"{i}. {self._track_label(t)}" for i, t in enumerate(tracks, start=1)
        )
        return (
            f"{propmt}<br>"
            f"{listing}<br>"
            "Pick a number, name the artist, or say <i>cancel</i>."
        )


    def _options_act(self, tracks: list[dict]) -> DialogueAct:
        def option(title, payload):
            return SlotValueAnnotation("option", json.dumps({"title": title, "payload": payload}, ensure_ascii=False))

        annotations = [option(self._plain_label(t), self._plain_label(t)) for t in tracks]
        annotations.append(option("Cancel", "cancel"))
        return DialogueAct(intent=_INTENT_OPTIONS, annotations=annotations)

    def _parse_choice(self, text: str, options: list[dict]):
        t = norm(text)
        if t in _CANCEL or t.startswith("none of"):
            return "cancel", None

        words = [w for w in t.split() if w not in _FILLER]
        if len(words) == 1:
            w = words[0]
            m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)?", w)
            index = int(m.group(1)) if m else (len(options) if w == "last" else _ORDINALS.get(w))
            if index and 1 <= index <= len(options):
                return "pick", options[index - 1]

        if words:
            hits = [
                o for o in options
                if set(words) <= set(norm(self._plain_label(o)).split())
            ]
            if len(hits) == 1:
                return "pick", hits[0]
        return None, None

    def _handle_choice(self, command: str, options: list[dict], is_command: bool) -> str | None:

        for t in options:
            if norm(command) == norm(self._plain_label(t)):
                self._pending = None
                return self._add_chosen(t)
        
        if is_command:
            self._pending = None
            return None

        action, track = self._parse_choice(command, options)
        if action == "pick" and track is not None:
            self._pending = None
            return self._add_chosen(track)
        
        if action == "cancel":
            self._pending = None
            return "Okay, I won't add anything."

        self._offer = options
        return (
            "I didn't catch which one. Pick a number from the list, name the"
            " artist, or say <i>cancel</i>."
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
        if track is not None:
            return self._add_chosen(track)

        matches = self._db.find_tracks(query, limit=10)
        exact = self._distinct_songs([m for m in matches if m["exact"]])

        if len(exact) == 1:
            return self._add_chosen(exact[0])
        if len(exact) > 1:
            return self._ask_which(
                f"Several songs are called '{html.escape(query)}'. Which one do you mean?", self._rank_for_playlist(exact)[:5],
            )

        guesses = self._fuzzy_guesses(query)
        if guesses and guesses[0][1] >= FUZZY_ACCEPT:
            track = guesses[0][0]
            note = f" (I read '{html.escape(query)}' as '{self._track_label(track)}')"
            return self._add_chosen(track, note=note)

        options = [t for t, sim in guesses[:3] if sim >= FUZZY_SUGGEST] or matches[:3]
        options = self._distinct_songs(options)
        if not options:
            return f"I couldn't find '{html.escape(query)}' in the catalog."

        return self._ask_which(
            f"I couldn't find an exact match for '{html.escape(query)}'. Did you mean one of these?", options
        )

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
        self._offer = None
        command = utterance.text.strip()
        lowered = command.lower()
        m_add, m_remove = _ADD_RE.match(command), _REMOVE_RE.match(command)

        choice = None
        if self._pending:
            is_command = bool(
                m_add or m_remove or lowered in _VIEW or lowered in _CLEAR or lowered.startswith("/") or is_help_request(command)

            )
            choice = self._handle_choice(command, self._pending, is_command)
        if choice is not None:
            response = choice

        # View playlist.
        elif lowered in _VIEW:
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

        elif is_help_request(command):
            response = help_answer(command)

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
                "I'm sorry, I don't understand that."
                "Say <i>help</i> to see what I can do."
            )
        dialogue_acts.append(self._playlist_act())
        if self._offer:
            dialogue_acts.append(self._options_act(self._offer))
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
    get_fuzzy_index(TrackDatabase())
    platform = FlaskSocketPlatform(MusicCRS)
    platform.start()

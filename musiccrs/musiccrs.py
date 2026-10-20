
"""MusicCRS conversational agent."""
import html
import json
import os
import re

from typing import Callable
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

from track_database import TrackDatabase, norm, clean_tags
from fuzzy import FuzzyTrackIndex
from capabilities import help_answer, is_help_request
from playlist_generator import PlaylistGenerator, warm_up as warm_up_generator
from questions import ARTIST_KINDS, IT_WORDS, Question, parse_question
from recommender import Recommender

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


_RECOMMENDER = None

def get_recommender(db):
    global _RECOMMENDER
    if _RECOMMENDER is None:
        _RECOMMENDER = Recommender(db.db_path)
    return _RECOMMENDER

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

_RECOMMEND_RE = re.compile(
    r"^(?:/recommend"
    r"|(?:(?:please|ok(?:ay)?)[, ]+)?(?:(?:can|could|would) you (?:please )?)?"
    r"(?:recommend|suggest)(?: me| us)?"
    r"(?: (?:something|some|a few|a couple of|a|more|another|new|other|3|4|5|three|four|five))*"
    r"(?: (?:songs?|tracks?|music|ones?|things?))?"
    r"(?: (?:for me|to add|please|like (?:these|this|them)|based on (?:my|the) playlist"
    r"|for (?:my|the) playlist|that fit (?:my|the) playlist))*"
    r"|what (?:should|could|can) i (?:add|listen to|play)(?: next)?"
    r"|(?:give me |show me )?(?:some |more )*(?:recommendations|suggestions)"
    r"|more (?:songs )?like (?:this|these|them))\s*[?!.]*$",
    re.I,
)

_GENERATE_RE = re.compile(
    r"^(?:/generate\b"
    r"|(?:(?:please|ok(?:ay)?)[, ]+)?(?:(?:can|could|would) you (?:please )?)?"
    r"(?:make|create|generate|build|put together|give)\s+(?:me\s+|us\s+)?(?:a|an|the|some)?\s*(?:new\s+)?"
    r"play ?list\b(?:\s+(?:for|of|with|about|that|to|full of|containing))?)"
    r"\s*(?P<d>.*?)\s*[.!?]*$",
    re.I,
)
_YES = {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "do it", "go ahead", "continue",
        "please do", "yes please", "replace it", "replace", "confirm"}
_NO = {"no", "n", "nope", "cancel", "nevermind", "never mind", "keep it", "dont", "no thanks",
       "no thank you", "keep my playlist", "keep the playlist", "stop"}


_VIEW = ("/playlist", "/list", "show playlist", "show my playlist", "view playlist", "what's in my playlist", "what is in my playlist")
_CLEAR = ("/clear", "clear playlist", "clear my playlist", "empty playlist")

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
_CANCEL = {"cancel", "no", "none", "neither", "nevermind", "never mind", "skip", "no thanks", "no thank you", "nope", "not now", "not this time", "stop", "quit", "exit", "leave", "dismiss", "forget it", "forget about it", "forget that", "forget this", "forget it all", "forget everything", "forget all of this", "forget everything I said", "forget everything I just said", "forget everything I just told you", "forget everything I just told you about my playlist", "forget everything I just told you about my music preferences"}
_FILLER = {"the", "one", "ones", "please", "i", "want", "pick", "choose", "select", "take", "song", "track", "by", "a", "with", "number", "option", "it", "that", "go", "for", "me", "my", "to", "in", "on", "of", "and", "or", "but", "so", "if", "then", "else", "when", "where", "how", "what", "which", "who", "whom", "whose", "why", "because", "since", "although", "though", "even though"}
_QUIT = {"quit", "exit", "bye", "goodbye", "good bye", "stop"}

class MusicCRS(Agent):
    def __init__(self, use_llm: bool = True):
        """Initialize MusicCRS agent."""
        self._playlist_generator = None
        self._confirm_generate: str | None = None
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
        self._pending_then: Callable[[dict, str], str] | None = None
        self._pending_cancel = "Okay, I won't add anything."
        self._last_track: dict | None = None
        self._recommended: list[str] = []

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
        self._last_track = track
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

    def _ask_which(self, propmt: str, tracks: list[dict], then: Callable[[dict, str], str] | None = None, cancel_text: str = "Okay, I won't add anything.", notes: list[str] | None = None) -> str:
        self._pending = tracks
        self._offer = tracks
        self._pending_then = then or self._add_chosen
        self._pending_cancel = cancel_text
        listing = "<br>".join(
            f"{i}. {self._track_label(t)}" + (f" <i> ({html.escape(notes[i-1])}) </i>" if notes else "") for i, t in enumerate(tracks, start=1)
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
        then = self._pending_then or self._add_chosen
        for t in options:
            if norm(command) == norm(self._plain_label(t)):
                self._pending = None
                return then(t, "")
        
        if is_command:
            self._pending = None
            return None

        action, track = self._parse_choice(command, options)
        if action == "pick" and track is not None:
            self._pending = None
            return then(track, "")
        
        if action == "cancel":
            self._pending = None
            return self._pending_cancel

        self._offer = options
        return (
            "I didn't catch which one. Pick a number from the list, name the"
            " artist, or say <i>cancel</i>."
        )
    # --------------------------------------------------
    # Add track
    # --------------------------------------------------

    def _lookup(self, query: str, then: Callable[[dict, str], str], cancel_text: str = "Okay, I won't add anything.") -> str:
        track = self._db.get_track(query)
        if track is not None:
            return then(track, "")
        matches = self._db.find_tracks(query, limit=10)
        exact = self._distinct_songs([m for m in matches if m["exact"]])
        if len(exact) == 1:
            return then(exact[0], "")
        if len(exact) > 1:
            return self._ask_which(
                f"Several songs are called '{html.escape(query)}'. Which one do you mean?", self._rank_for_playlist(exact)[:5], then, cancel_text,
            )

        guesses = self._fuzzy_guesses(query)
        if guesses and guesses[0][1] >= FUZZY_ACCEPT:
            track = guesses[0][0]
            note = f" (I read '{html.escape(query)}' as '{self._track_label(track)}')"
            return then(track, note)

        options = [t for t, sim in guesses[:3] if sim >= FUZZY_SUGGEST] or matches[:3]
        options = self._distinct_songs(options)
        if not options:
            return f"I couldn't find '{html.escape(query)}' in the catalog."

        return self._ask_which(
            f"I couldn't find an exact match for '{html.escape(query)}'. Did you mean one of these?", options, then, cancel_text
        )

    def _add_track(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "Please provide a track name."
        return self._lookup(query, self._add_chosen)

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

    @staticmethod
    def _plural(n: int, word: str) -> str:
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    def _answer_question(self, q: Question) -> str:
        subject = q.subject

        if norm(subject) in IT_WORDS:
            if q.kind in ARTIST_KINDS:
                return "Please name the artist, for example: <i>what songs does Coldplay have?</i>"
            if self._last_track is None:
                return "Which song do you mean? For example: <i>who sings Yellow?</i>"

            return self._answer_track(q.kind, self._last_track)   

        if q.kind in ARTIST_KINDS:
            return self._artist_question(q.kind, subject)

        if q.kind in ("tags", "about"):
            artists = self._db.find_artists(subject, limit=1)
            if artists and artists[0]["similarity"] == 1.0:
                return self._answer_artist(q.kind, artists[0])

        if q.kind == "track_artist":
            songs = self._distinct_songs([m for m in self._db.find_tracks(subject, limit=10) if m["exact"]])
            if len(songs) > 1:
                listing = "<br>".join(
                    self._track_label(t) for t in self._rank_for_playlist(songs)[:5]
                )
                return f"Several songs are called '{html.escape(subject)}':<br>{listing}"
        return self._lookup(subject, lambda track, note: self._answer_track(q.kind, track) + note, cancel_text="Okay, never mind.")

    def _artist_question(self, kind: str, subject: str) -> str:
        found = self._db.find_artists(subject, limit=3)
        if not found:
            return f"I couldn't find an artist called '{html.escape(subject)}'."
        best = found[0]
        if best["similarity"] < 0.85:
            names = ", ".join(f"<b>{html.escape(a['name'])}</b>" for a in found)
            return ( 
                f"I couldn't find an artist called '{html.escape(subject)}'. Did you mean {names}?"
            )
        note = ""
        if best ["similarity"] < 1.0:
            note = f" (I read '{html.escape(subject)}' as '{html.escape(best['name'])}')"
        return self._answer_artist(kind, best) + note

    
    def _answer_track(self, kind: str, track: dict) -> str:
        self._last_track = track
        d = self._db.get_details(track["track_id"]) or track
        title = f"<b>{html.escape(d['track_name'])}</b>"
        by = html.escape(", ".join(d["artist_name"]))
        date = d.get("release_date")
        ms = d.get("duration")
        pop = d.get("popularity") or 0
        album = html.escape(", ".join(d["album_name"]))
        length = f"{ms // 60000}:{ms // 1000 % 60:02d}" if ms else None

        if kind == "track_artist":
            return f"{title} is by <b>{by}</b>."
        if kind == "track_album":
            albums = [html.escape(a) for a in d["album_name"]]
            if not albums:
                return f"I don't have album information for {title} by {by}."
            if len(albums) == 1:
                return f"{title} by {by} is on the album <b>{albums[0]}</b>."
            names = " and ".join(f"<b>{a}</b>" for a in albums[:2])
            more = f" (and {len(albums) - 2} more)" if len(albums) > 2 else ""
            return f"{title} by {by} appears on the albums {names}{more}."
        if kind == "track_year":
            if not date:
                return f"I don't have a release date for {title} by {by}."
            return f"{title} by {by} was released in <b>{date[:4]}</b> ({date})."
        if kind == "track_length":
            if not length:
                return f"I don't have the length of {title} by {by}."
            return f"{title} by {by} is <b>{length}</b> long."
        if kind == "track_popularity":
            if not pop:
                return f"I don't have a popularity score for {title} by {by}."
            pct = min(99, int(self._db.popularity_percentile(pop)))
            return (
                f"{title} by {by} has a popularity score of <b>{pop:.0f}/100</b>, "
                f"higher than {pct}% of the songs in my catalog."
            )
        if kind == "tags":
            tags = clean_tags(d.get("tags") or [])
            if not tags:
                return f"I don't have tags for {title} by {by}."
            return f"{title} by {by} is tagged: {html.escape(', '.join(tags))}."

        parts = [f"{title} is by <b>{by}</b>"]                # "about"
        if d["album_name"]:
            parts.append(f"on the album {html.escape(d['album_name'][0])}")
        if date:
            parts.append(f"released in {date[:4]}")
        if length:
            parts.append(f"{length} long")
        if pop:
            parts.append(f"popularity {pop:.0f}/100")
        return ", ".join(parts) + "."

    def _answer_artist(self, kind: str, artist: dict) -> str:
        name = f"<b>{html.escape(artist['name'])}</b>"
        n = artist["tracks"]
        songs = self._plural(n, "song")

        if kind == "artist_count":
            return f"I have {songs} by {name}."
        if kind == "artist_songs":
            top = self._db.artist_tracks(artist["key"], limit=5)
            listing = "<br>".join(
                f"{i}. {html.escape(t['track_name'])}" for i, t in enumerate(top, start=1)
            )
            lead = "The most popular" if n > len(top) else "They are"
            return f"I have {songs} by {name}. {lead}:<br>{listing}"
        if kind == "artist_albums":
            albums = self._db.artist_albums(artist["key"])
            if not albums:
                return f"I don't have album information for {name}."
            return f"Albums by {name} in my catalog: {html.escape(', '.join(albums))}."
        if kind == "artist_top":
            top = self._db.artist_tracks(artist["key"], limit=1)
            pop = f" (popularity {top[0]['popularity']:.0f}/100)" if top and top[0]["popularity"] else ""
            return f"The most popular song by {name} is <b>{html.escape(top[0]['track_name'])}</b>{pop}."
        if kind == "tags":
            tags = self._db.artist_tags(artist["key"])
            if not tags:
                return f"I don't have style information for {name}."
            return f"{name} is mostly tagged: {html.escape(', '.join(tags))}."

        top = self._db.artist_tracks(artist["key"], limit=3)   # "about"
        text = f"{name}: {songs} in my catalog"
        if top:
            text += ". Most popular: " + html.escape(", ".join(t["track_name"] for t in top))
        tags = self._db.artist_tags(artist["key"], k=4)
        if tags:
            text += ". Known for: " + html.escape(", ".join(tags))
        return text + "."

    # --------------------------------------------------
    # R8: playlist generation
    # --------------------------------------------------

    def _generate_playlist(self, description: str, confirmed: bool = False) -> str:
        """Generate and replace the playlist from a description."""

        description = description.strip()

        if not description:
            return (
                "Please describe the playlist you want. "
                "For example: /generate energetic music for the gym"
            )

        if self._llm is None:
            return "Playlist generation requires an LLM."

        if self._playlist and not confirmed:
            self._confirm_generate = description
            return (
                f"A new playlist for '{html.escape(description)}' <b>replaces your current playlist<b/> of {len(self._playlist)} songs. "
                "Say <b>yes<b/> to continue or <b>no</b> to keep it."
            )

        try:
            if self._playlist_generator is None:
                self._playlist_generator = PlaylistGenerator(self._llm)

            result = self._playlist_generator.generate(description)

            # Only accept track IDs that exist in our SQLite database.
            verified = self._db.get_tracks(result["track_ids"])
            track_ids = list(dict.fromkeys(t["track_id"] for t in verified))

            if not track_ids:
                return (
                    "I couldn't find suitable songs for that description. "
                    "Please try describing a different mood or genre."
                )

            # Replace the existing playlist only after successful generation.
            self._playlist = track_ids
            self._recommended.clear()

            # Clear any outstanding song-selection state.
            self._pending = None
            self._offer = None

            count = len(track_ids)
            minutes = result["actual_minutes"]

            return (
                f"Created a new playlist based on "
                f"'{html.escape(description)}'!<br>"
                f"<b>{count} songs</b>, approximately "
                f"<b>{minutes:.0f} minutes</b> long.<br>"
                "Your playlist has been updated."
            )

        except Exception:
            import logging
            logging.exception("R8 playlist generation failed")
            return (
                "Sorry, I couldn't generate your playlist right now. "
                "Please try again."
            )

    def _handle_confirmation(self, command: str) -> str | None:
        if self._confirm_generate is None:
            return None
        description, self._confirm_generate = self._confirm_generate, None
        answer = norm(command)
        if answer in _YES:
            return self._generate_playlist(description, confirmed=True)
        if answer in _NO:
            return "Okay, I'll keep your current playlist."
        return None

    def _recommend(self) -> str:
        if not self._playlist:
            return(
                "Your playlist is empty, so I have nothing to base a recommendation on. "
                "Add a few songs first, for example: <i>add Yellow by Coldplay</i>."
            )
        picks = get_recommender(self._db).recommend(self._playlist, k=5, skip=self._recommended)
        tracks = {t["track_id"]: t for t in self._db.get_tracks([i for i, _ in picks])}
        picks = [(i, why) for i, why in picks if i in tracks]
        if not picks:
            return "I couldn't find more songs that fit your playlist."
        self._recommended.extend(i for i, _ in picks)
        return self._ask_which(
            "Based on your playlist, you might like these:", [tracks[i] for i, _ in picks], cancel_text="Okay, I won't add any of them. Say <i>recommend more</i> for other songs", notes=[why for _, why in picks]
        )
    # --------------------------------------------------

    def receive_utterance(self, utterance: Utterance) -> None:
        """Handle a new user message."""
        dialogue_acts = []
        self._offer = None
        command = utterance.text.strip()
        lowered = command.lower()
        m_add, m_remove = _ADD_RE.match(command), _REMOVE_RE.match(command)
        question = parse_question(command)
        m_generate = _GENERATE_RE.match(command)
        confirmation = self._handle_confirmation(command)

        choice = None
        if self._pending and confirmation is None:
            is_command = bool(
                m_add or m_remove or lowered in _VIEW or lowered in _CLEAR or lowered.startswith("/") or is_help_request(command)
                or question is not None or _RECOMMEND_RE.match(command) or _GENERATE_RE.match(command)
            )
            choice = self._handle_choice(command, self._pending, is_command)
        if confirmation is not None:
            response = confirmation
        
        elif choice is not None:
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
            self._recommended.clear()
            response = "Your playlist has been cleared."

        # R8: Generate a complete playlist.
        elif m_generate:
            response = self._generate_playlist(m_generate.group("d"))

        # Add a verified track.
        elif m_add:
            response = self._add_track(m_add.group("q"))

        # Remove a track.
        elif m_remove:
            response = self._remove_track(m_remove.group("q"))

        elif is_help_request(command):
            response = help_answer(command)

        elif _RECOMMEND_RE.match(command):
            response = self._recommend()

        elif question is not None:
            response = self._answer_question(question)

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
        elif lowered == "/quit" or norm(command) in _QUIT:
            self.goodbye()
            return

        else:
            response = (
                "I'm sorry, I don't understand that. "
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
    get_recommender(TrackDatabase())
    warm_up_generator()
    platform = FlaskSocketPlatform(MusicCRS)
    platform.start()

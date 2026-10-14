"""MusicCRS conversational agent."""

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.platforms import FlaskSocketPlatform
from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv


load_dotenv()


LLM_BASE_URL = "https://openwebui.ux.uis.no/api"
LLM_MODEL = "gorina10.gemma-4-31b"
LLM_API_KEY = os.getenv("MY_OLLAMA_API_KEY")  # Load from .env or environment variable



_INTENT_OPTIONS = Intent("OPTIONS")


class MusicCRS(Agent):
    def __init__(self, use_llm: bool = True):
        """Initialize MusicCRS agent."""
        super().__init__(id="MusicCRS")

        if use_llm:
            self._llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        else:
            self._llm = None

        self._playlist = []  # Stores the current playlist

    def welcome(self) -> None:
        """Sends the agent's welcome message."""
        utterance = AnnotatedUtterance(
            "Hello, I'm MusicCRS. What are you in the mood for?",
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def goodbye(self) -> None:
        """Quits the conversation."""
        utterance = AnnotatedUtterance(
            "It was nice talking to you. Bye",
            dialogue_acts=[DialogueAct(intent=self.stop_intent)],
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def _playlist_response(self, message: str) -> str:
        """Return the current playlist state to the frontend."""
        return "__PLAYLIST_STATE__" + json.dumps({
            "message": message,
            "tracks": self._playlist,
        })

    def receive_utterance(self, utterance: Utterance) -> None:
        """Gets called each time there is a new user utterance.

        For now the agent only understands specific command.

        Args:
            utterance: User utterance.
        """
        response = ""
        dialogue_acts = []
        command = utterance.text.strip()
        lowered = command.lower()
        if lowered in ("/playlist", "show playlist", "show my playlist", "view playlist", "what is in my playlist", "what's in my playlist"):
            if not self._playlist:
                response = self._playlist_response("Your playlist is empty.")
            else:
                tracks = "\n".join(
                    f"{i}. {track}"
                    for i, track in enumerate(self._playlist, start=1)
                )

                response = self._playlist_response(
                    f"Your playlist contains {len(self._playlist)} songs:\n{tracks}"
                )
        elif lowered in ("/clear", "clear playlist", "clear my playlist", "empty playlist"):
            self._playlist.clear()
            response = self._playlist_response("Playlist cleared.")
        elif lowered.startswith("/add ") or lowered.startswith("add "):
            name = re.sub(r"^/?add\s+", "", command, flags=re.IGNORECASE).strip()
            if not name:
                response = self._playlist_response("Please provide a track name.")
            elif name.casefold() in (track.casefold() for track in self._playlist):
                response = self._playlist_response(f"{name} is already in your playlist.")
            else:
                self._playlist.append(name)
                response = self._playlist_response(f"Added {name}.")
        elif lowered.startswith("/remove ") or lowered.startswith("remove "):
            name = re.sub(r"^/?remove\s+", "", command, flags=re.IGNORECASE).strip()
            found = next((t for t in self._playlist if t.casefold() == name.casefold()), None)
            if found is None:
                response = self._playlist_response(f"Could not find {name} in your playlist.")
            else:
                self._playlist.remove(found)
                response = self._playlist_response(f"Removed {found}.")        
        elif utterance.text.startswith("/info"):
            response = self._info()
        elif utterance.text.startswith("/ask_llm "):
            prompt = utterance.text[9:]
            response = self._ask_llm(prompt)
        elif utterance.text.startswith("/options"):
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
                        SlotValueAnnotation("option", option) for option in options
                    ],
                )
            ]
        elif utterance.text == "/quit":
            self.goodbye()
            return
        else:
            response = "I'm sorry, I don't understand that command."

        self._dialogue_connector.register_agent_utterance(
            AnnotatedUtterance(
                response,
                participant=DialogueParticipant.AGENT,
                dialogue_acts=dialogue_acts,
            )
        )

    # --- Response handlers ---

    def _info(self) -> str:
        """Gives information about the agent."""
        return "I am MusicCRS, a conversational recommender system for music."

    def _ask_llm(self, prompt: str) -> str:
        """Calls a large language model (LLM) with the given prompt.

        Args:
            prompt: Prompt to send to the LLM.

        Returns:
            Response from the LLM.
        """
        if not self._llm:
            return "The agent is not configured to use an LLM"

        llm_response = self._llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # optional: controls randomness
        )

        return f"LLM response: {llm_response.choices[0].message.content}"

    def _options(self, options: list[str]) -> str:
        """Presents options to the user."""
        return (
            "Here are some options:\n<ol>\n"
            + "\n".join([f"<li>{option}</li>" for option in options])
            + "</ol>\n"
        )


if __name__ == "__main__":
    platform = FlaskSocketPlatform(MusicCRS)
    platform.start()

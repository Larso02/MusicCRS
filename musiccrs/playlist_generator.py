
"""Automatic playlist generation for MusicCRS (R8)."""
import json
import math
import re
import sys
from pathlib import Path

# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25 import BM25Retriever
from retrieval.data_loader import MusicCatalogLoader

class PlaylistGenerator:
    """Generate playlists from natural-language descriptions."""

    def __init__(self, llm):
        self.llm = llm

        # Index music-related metadata, including descriptive tags.
        self.retriever = BM25Retriever(
            corpus_types=[
                "track_name",
                "artist_name",
                "album_name",
                "tag_list",
            ]
        )

        # Access complete track metadata, including actual duration.
        self.catalog = MusicCatalogLoader()
        self.metadata = self.catalog.metadata_dict

    def interpret_request(self, description: str) -> dict:
        """Use the LLM to interpret the playlist request."""

        prompt = f"""
You are a music playlist planning assistant.

Analyze the user's request and return ONLY a JSON object.

Required JSON format:
{{
    "search_query": "descriptive music keywords",
    "duration_minutes": 60,
    "explanation": "brief reason for the duration"
}}

Rules:
- Search query should contain relevant genres, moods,
  instruments, or music styles.
- Estimate an appropriate playlist duration based on context.
- Do NOT ask the user for the number of songs.
- Short activities: approximately 15-30 minutes.
- General listening: approximately 40-60 minutes.
- Gym sessions: approximately 45-75 minutes.
- Studying: approximately 60-120 minutes.
- Road trips: approximately 90-180 minutes.
- Return valid JSON only.

User request: {description}
"""

        response = self.llm.chat.completions.create(
            model="gorina10.gemma-4-31b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()

        # Allow JSON enclosed in a Markdown code block.
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
        ).strip()

        return json.loads(content)

    @staticmethod
    def duration_ms(track: dict) -> int:
        """Extract track duration in milliseconds."""
        value = track.get("duration", 0)

        if isinstance(value, list):
            value = value[0] if value else 0

        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def generate(self, description: str) -> dict:
        """Generate a complete playlist based on a description."""

        if not description.strip():
            raise ValueError("Please provide a playlist description.")

        plan = self.interpret_request(description)

        search_query = str(
            plan.get("search_query", description)
        ).strip()

        try:
            target_minutes = float(
                plan.get("duration_minutes", 45)
            )
        except (ValueError, TypeError):
            target_minutes = 45

        if not math.isfinite(target_minutes):
            target_minutes = 45

        # Prevent excessively short or long playlists.
        target_minutes = max(15, min(target_minutes, 120))
        target_ms = target_minutes * 60 * 1000

        # Retrieve a reasonably large candidate pool.
        candidate_ids = self.retriever.text_to_item_retrieval(
            search_query,
            topk=200,
        )

        selected = []
        selected_ids = set()
        artist_counts = {}
        total_ms = 0

        for track_id in candidate_ids:
            track = self.metadata.get(track_id)

            if not track or track_id in selected_ids:
                continue

            duration = self.duration_ms(track)

            # Exclude missing or implausible durations.
            if not 60_000 <= duration <= 900_000:
                continue

            artists = track.get("artist_name", [])
            artist = (
                str(artists[0]).casefold()
                if artists
                else "unknown"
            )

            # Allow at most 3 tracks by the same primary artist.
            if artist_counts.get(artist, 0) >= 3:
                continue

            selected.append(track_id)
            selected_ids.add(track_id)

            artist_counts[artist] = (
                artist_counts.get(artist, 0) + 1
            )

            total_ms += duration

            if total_ms >= target_ms or len(selected) >= 40:
                break

        return {
            "track_ids": selected,
            "description": description,
            "search_query": search_query,
            "target_minutes": target_minutes,
            "actual_minutes": round(total_ms / 60_000, 1),
            "explanation": plan.get("explanation", ""),
        }

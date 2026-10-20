
"""Test R8 playlist generation."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from musiccrs.playlist_generator import PlaylistGenerator
from musiccrs.track_database import TrackDatabase


load_dotenv()

llm = OpenAI(
    api_key=os.getenv("MY_OLLAMA_API_KEY"),
    base_url="https://openwebui.ux.uis.no/api",
)

generator = PlaylistGenerator(llm)
db = TrackDatabase()

descriptions = [
    "sad love songs",
    "energetic music for a gym session",
    "relaxing instrumental music for studying",
]

for description in descriptions:
    print(f"\nREQUEST: {description}")

    result = generator.generate(description)

    tracks = db.get_tracks(result["track_ids"])

    print("Search query:", result["search_query"])
    print("Target duration:", result["target_minutes"], "minutes")
    print("Actual duration:", result["actual_minutes"], "minutes")
    print("Number of songs:", len(tracks))

    for i, track in enumerate(tracks, start=1):
        artists = ", ".join(track["artist_name"])
        print(f"{i}. {track['track_name']} — {artists}")

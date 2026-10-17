
"""Import the TalkPlay track catalog into a local SQLite database."""

import json
import sqlite3
from pathlib import Path

from retrieval.data_loader import MusicCatalogLoader
from musiccrs.track_database import norm


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "tracks.db"

def as_list(value):
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)

def main():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading TalkPlay track metadata...")
    catalog = MusicCatalogLoader()

    print(f"Loaded {len(catalog.metadata_dict)} tracks.")

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS tracks")
        conn.execute("""
            CREATE TABLE tracks (
                track_id TEXT PRIMARY KEY,
                track_name TEXT NOT NULL,
                artist_name TEXT NOT NULL,
                album_name TEXT,
                popularity REAL DEFAULT 0,
                title_norm TEXT NOT NULL,
                artist_norm TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX idx_tracks_title_norm ON tracks(title_norm)
        """)

        rows = []

        for track_id, metadata in catalog.metadata_dict.items():
            artists = as_list(metadata.get("artist_name"))
            albums = as_list(metadata.get("album_name"))
            titles = as_list(metadata.get("track_name"))
            title = titles[0] if titles else ""


            rows.append((
                str(track_id),
                title,
                json.dumps(artists, ensure_ascii=False),
                json.dumps(albums, ensure_ascii=False),
                metadata.get("popularity", 0),
                norm(title),
                norm(" ".join(artists)),

            ))


        conn.executemany("""
            INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)

        count = conn.execute(
            "SELECT COUNT(*) FROM tracks"
        ).fetchone()[0]

    print(f"Successfully imported {count} tracks.")
    print(f"Database saved to: {DATABASE_PATH}")


if __name__ == "__main__":
    main()

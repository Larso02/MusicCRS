
"""Import the TalkPlay track catalog into a local SQLite database."""

import json
import sqlite3
from pathlib import Path

from retrieval.data_loader import MusicCatalogLoader


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "tracks.db"


def main():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading TalkPlay track metadata...")
    catalog = MusicCatalogLoader()

    print(f"Loaded {len(catalog.metadata_dict)} tracks.")

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id TEXT PRIMARY KEY,
                track_name TEXT NOT NULL,
                artist_name TEXT NOT NULL,
                album_name TEXT,
                popularity REAL DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tracks_name
            ON tracks(track_name COLLATE NOCASE)
        """)

        rows = []

        for track_id, metadata in catalog.metadata_dict.items():
            artists = metadata.get("artist_name") or []
            albums = metadata.get("album_name") or []

            if isinstance(artists, str):
                artists = [artists]

            if isinstance(albums, str):
                albums = [albums]

            
            titles = metadata.get("track_name") or []

            if isinstance(titles, str):
              titles = [titles]

            # Use the first title as the primary display name.
            primary_title = titles[0] if titles else ""

            rows.append((
                str(track_id),
                primary_title,
                json.dumps(artists, ensure_ascii=False),
                json.dumps(albums, ensure_ascii=False),
                metadata.get("popularity") or 0,
            ))


        conn.executemany("""
            INSERT OR REPLACE INTO tracks
                (track_id, track_name, artist_name, album_name, popularity)
            VALUES (?, ?, ?, ?, ?)
        """, rows)

        count = conn.execute(
            "SELECT COUNT(*) FROM tracks"
        ).fetchone()[0]

    print(f"Successfully imported {count} tracks.")
    print(f"Database saved to: {DATABASE_PATH}")


if __name__ == "__main__":
    main()

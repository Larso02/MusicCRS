
"""Import the TalkPlay track catalog into a local SQLite database."""

import json
import sqlite3
from pathlib import Path

from retrieval.data_loader import MusicCatalogLoader
from musiccrs.track_database import norm, artist_key


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "tracks.db"

def as_list(value):
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)

def clean_date(value):
    value = (value or "").strip()
    return value if len(value) >= 4 and value[:4].isdigit() and int(value[:4]) > 1900 else None


def to_int(value):
    return int(value) if str(value).isdigit() else None

def main():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading TalkPlay track metadata...")
    catalog = MusicCatalogLoader()

    print(f"Loaded {len(catalog.metadata_dict)} tracks.")

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS tracks")
        conn.execute("DROP TABLE IF EXISTS track_artists")
        conn.execute("""
            CREATE TABLE tracks (
                track_id TEXT PRIMARY KEY,
                track_name TEXT NOT NULL,
                artist_name TEXT NOT NULL,
                album_name TEXT,
                popularity REAL DEFAULT 0,
                title_norm TEXT NOT NULL,
                artist_norm TEXT NOT NULL,
                release_date Text,
                duration INTEGER,
                tags TEXT
            )
        """)

        conn.execute("""
            CREATE INDEX idx_tracks_title_norm ON tracks(title_norm)
        """)
        conn.execute("""
            CREATE TABLE track_artists (
                artist_key TEXT NOT NULL,
                artist_name TEXT NOT NULL,
                track_id TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX idx_track_artists_artist_key ON track_artists(artist_key)")

        rows, links = [], []

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
                clean_date(metadata.get("release_date")),
                to_int(metadata.get("duration")),
                json.dumps(as_list(metadata.get("tag_list")), ensure_ascii=False),

            ))
            for artist in dict.fromkeys(artists):
                links.append((artist_key(artist), artist, str(track_id)))


        conn.executemany("""
            INSERT INTO tracks (track_id, track_name, artist_name, album_name, popularity, title_norm, artist_norm, release_date, duration, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

        conn.executemany("""
            INSERT INTO track_artists VALUES (?, ?, ?)
        """, links)

        count = conn.execute(
            "SELECT COUNT(*) FROM tracks"
        ).fetchone()[0]
        artists = conn.execute("SELECT COUNT(DISTINCT artist_key) FROM track_artists").fetchone()[0]

    print(f"Successfully imported {count} tracks by {artists} artists.")
    print(f"Database saved to: {DATABASE_PATH}")


if __name__ == "__main__":
    main()

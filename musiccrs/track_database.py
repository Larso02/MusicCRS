
"""SQLite track lookup for the TalkPlay music catalog."""

import json
import sqlite3
from pathlib import Path

#from matplotlib.pyplot import title
#from sympy import limit


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "tracks.db"


class TrackDatabase:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = Path(db_path)

        if not self.db_path.is_file():
            raise FileNotFoundError(
                f"Track database not found: {self.db_path}. "
                "Run: python -m scripts.import_tracks"
            )

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _format_track(row):
        return {
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "artist_name": json.loads(row["artist_name"]),
            "album_name": json.loads(row["album_name"] or "[]"),
            "popularity": row["popularity"],
        }

    def get_track(self, track_id):
        """Retrieve a track by its unique database ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tracks WHERE track_id = ?",
                (track_id,),
            ).fetchone()

        return self._format_track(row) if row else None


    def search_tracks(self, title, artist=None, limit=10):
        """Search by title and optionally artist, prioritizing exact matches."""
        title = title.strip()

        if not title:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM tracks
                WHERE track_name LIKE ? COLLATE NOCASE
                ORDER BY
                    CASE
                        WHEN track_name = ? COLLATE NOCASE THEN 0
                        ELSE 1
                    END,
                    popularity DESC
                """,
                (f"%{title}%", title),
            ).fetchall()

        tracks = [self._format_track(row) for row in rows]

        if artist:
            artist_query = artist.casefold().strip()
            tracks = [
                track for track in tracks
                if any(
                    artist_query in name.casefold()
                    for name in track["artist_name"]
                )
            ]

        return tracks[:limit]


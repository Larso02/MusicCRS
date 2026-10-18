
"""SQLite track lookup for the TalkPlay music catalog."""

import json
import sqlite3
from pathlib import Path
import re
import unicodedata
from contextlib import closing

#from matplotlib.pyplot import title
#from sympy import limit


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "tracks.db"



def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"['’`]", "", text)
    return re.sub(r"[\W_]+", " ", text).strip()
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
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM tracks WHERE track_id = ?",
                (track_id,),
            ).fetchone()

        return self._format_track(row) if row else None

    def get_tracks(self, track_ids):
        if not track_ids:
            return []
        marks = ",".join("?" * len(track_ids))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM tracks WHERE track_id IN ({marks})",
                track_ids,
            ).fetchall()
        by_id = {row["track_id"]: self._format_track(row) for row in rows}
        return [by_id[t] for t in track_ids if t in by_id]

    def all_norms(self):
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT track_id, title_norm, artist_norm, popularity FROM tracks"
            ).fetchall()

    def search_tracks(self, title, artist=None, limit=10):
        """Search by title and optionally artist, prioritizing exact matches."""
        t = norm(title)

        if not t:
            return []
        a = norm(artist) if artist else ""

        with closing(self._connect()) as conn:
            exact = conn.execute(
                """SELECT * FROM tracks WHERE title_norm = ? AND artist_norm LIKE ?
                ORDER BY popularity DESC LIMIT ?""",
                (t, f"%{a}%", limit),
            ).fetchall()
            partial = conn.execute(
                """SELECT * FROM tracks WHERE title_norm LIKE ? AND title_norm != ?
                  AND artist_norm LIKE ?
                ORDER BY popularity DESC LIMIT ?""",
                (f"%{t}%",t, f"%{a}%", limit),
            ).fetchall()

        results = [{**self._format_track(row), "exact": True} for row in exact]
        results += [{**self._format_track(row), "exact": False} for row in partial]

        return results[:limit]

    def find_tracks(self, text, limit=10):
        text = text.strip()
        hits = self.search_tracks(text, limit=limit)
        if any(hit["exact"] for hit in hits):
            return hits

        for m in reversed(list(re.finditer(r"\s+by\s+", text, flags=re.I))):
            found = self.search_tracks(text[: m.start()], text[m.end() :], limit=limit)
            if found:
                return found

        if ":" in text:
            artist, title = text.split(":", 1)
            found = self.search_tracks(title, artist, limit)
            if found:
                return found
        return hits    


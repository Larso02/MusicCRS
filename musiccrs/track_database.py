
"""SQLite track lookup for the TalkPlay music catalog."""

import json
import sqlite3
from pathlib import Path
import re
import unicodedata
from contextlib import closing
import difflib
from collections import Counter
from functools import lru_cache

#from matplotlib.pyplot import title
#from sympy import limit


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "tracks.db"

TAG_STOPLIST = {"favorites", "favorite", "favourites", "favourite", "my favorites", "awesome",
                "seen live", "good", "great", "amazing", "cool", "best", "check out"}

def artist_key(artist: str) -> str:
    return re.sub(r"^the ", "", norm(artist))

def clean_tags(tags, k: int = 6) -> list[str]:
    seen, out = set(), []
    for tag in tags:
        low = tag.strip().lower()
        if low and low not in seen and low not in TAG_STOPLIST:
            seen.add(low)
            out.append(tag.strip())
            if len(out) == k:
                break
    return out

@lru_cache(maxsize=4)
def _artist_table(db_path: str) -> dict:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute("SELECT artist_key, MIN(artist_name), COUNT(*) FROM track_artists GROUP BY artist_key").fetchall()
    return {key: (name, n) for key, name, n in rows}

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

    def get_details(self, track_id):
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM tracks WHERE track_id = ?", (track_id,)
            ).fetchone()
            if row is None:
                return None
            track = self._format_track(row)
            track["release_date"] = row["release_date"]
            track["duration"] = row["duration"]
            track["tags"] = json.loads(row["tags"] or "[]")
        return track

    def popularity_percentile(self, popularity):
        with closing(self._connect()) as conn:
            below, total = conn.execute(
                "SELECT SUM(popularity < ?), COUNT(*) FROM tracks WHERE popularity > 0" , (popularity,)
            ).fetchone()
        return 100.0* (below or 0) / total if total else 0.0

    def find_artists(self, text, limit=3):
        table = _artist_table(str(self.db_path))
        key = artist_key(text)
        if key in table:
            keys = [key]
        else:
            keys = difflib.get_close_matches(key, table, n=limit, cutoff=0.75)
        return [
            {
                "key": k,
                "name": table[k][0],
                "tracks": table[k][1],
                "similarity": difflib.SequenceMatcher(None, key, k).ratio(),
            }
            for k in keys
        ]

    def artist_tracks(self, key, limit=None):
        sql = (
            "SELECT t.* FROM tracks t JOIN track_artists a ON a.track_id = t.track_id"
            " WHERE a.artist_key = ? ORDER BY t.popularity DESC"
        )
        params = [key]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._format_track(row) for row in rows]

    def artist_albums(self, key, limit=8):
        best = {}
        for t in self.artist_tracks(key):
            for album in t["album_name"][:1]:
                best[album] = max(best.get(album, 0), t["popularity"])
        return sorted(best, key=lambda album: best[album], reverse=True)[:limit]
    

    def artist_tags(self, key, k=6):
        sql = (
            "SELECT t.tags FROM tracks t JOIN track_artists a ON a.track_id = t.track_id"
            " WHERE a.artist_key = ?"
        )
        counts = Counter()
        with closing(self._connect()) as conn:
            for (tags,) in conn.execute(sql, (key,)):
                counts.update({t.strip().lower() for t in json.loads(tags or "[]")} - TAG_STOPLIST - {""})
        return [tag for tag, _ in counts.most_common(k)]
import json
import sqlite3
from contextlib import closing

import numpy as np
import scipy.sparse as sp

try:
    from track_database import TAG_STOPLIST
except ImportError:
    from musiccrs.track_database import TAG_STOPLIST

WEIGHTS = {"artist": 1.0, "album": 0.5, "last": 1.0, "pop": 1.0, "tag": 1.0, "year":0.5}
MAX_PER_ARTIST = 2
MAX_KNOWN_ARTIST = 2

class Recommender:
    def __init__(self, db_path):
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute("SELECT track_id, artist_name, album_name, popularity, release_date, tags FROM tracks").fetchall()
            self.ids = [r[0] for r in rows]
            self.index = {t: i for i, t in enumerate(self.ids)}
            self.pop = np.array([r[4] or 0 for r in rows], dtype=np.float32)/100.0
            self.year = np.array([int(r[5][:4]) if r[5] and r[5][:4].isdigit() else 0 for r in rows], dtype=np.int32)
            self.titles = [r[1] for r in rows]
            artists, albums = {}, {}
            self.artist_of = np.array([artists.setdefault(r[2], len(artists)) for r in rows])
            self.album_of = np.array([albums.setdefault(r[3], len(albums)) for r in rows])

            vocab, r_idx, c_idx = {}, [], []
            self.tags: list[set[str]] = []
            for i, r in enumerate(rows):
                tags = {t.strip().lower() for t in json.loads(r[6] or "[]")} - TAG_STOPLIST - {""}
                self.tags.append(tags)
                for tag in tags:
                    r_idx.append(i)
                    c_idx.append(vocab.setdefault(tag, len(vocab)))
            self.vocab = vocab
            n = len(rows)
            m = sp.csr_matrix(
                (np.ones(len(r_idx), np.float32), (r_idx, c_idx)), shape=(n, max(len(vocab), 1))
            )
            df = np.array(m.sum(axis=0)).ravel()
            self.idf = np.log(n / (1 + df)).astype(np.float32)
            m = m.multiply(self.idf[None, :]).tocsr()
            norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
            norms[norms == 0] = 1.0
            self.tag_matrix = (sp.diags(1.0 / norms) @ m).tocsr()

    def scores(self, playlist_ids: list[str]) -> np.ndarray:
        h = np.array([self.index[t] for t in playlist_ids if t in self.index], dtype=int)
        w = WEIGHTS
        s = w["pop"] * self.pop
        s = s + w["artist"] * np.isin(self.artist_of, self.artist_of[h])
        s = s + w["album"] * np.isin(self.album_of, self.album_of[h])
        s = s + w["last"] * (self.artist_of == self.artist_of[h[-1]])

        centroid = np.asarray(self.tag_matrix[h].mean(axis=0)).ravel()
        norm = np.linalg.norm(centroid)
        if norm > 0:
            s = s + w["tag"] * (self.tag_matrix @ centroid / norm)

        years = self.year[h][self.year[h] > 0]
        if len(years):
            s = s + w["year"] * np.exp(-np.abs(self.year - years.mean()) / 8) * (self.year > 0)
        return s

    def recommend(self, playlist_ids: list[str], k: int = 5, skip=()) -> list[tuple[str, str]]:
        if not any(t in self.index for t in playlist_ids):
            return []

        s = self.scores(playlist_ids)
        blocked = [self.index[t] for t in (*playlist_ids, *skip) if t in self.index]
        s[blocked] = -1e9

        h = [self.index[t] for t in playlist_ids if t in self.index]
        list_artists = {self.artist_of[i] for i in h}
        list_albums = {self.album_of[i] for i in h}
        list_tags: dict[str, int] = {}
        for i in h:
            for tag in self.tags[i]:
                list_tags[tag] = list_tags.get(tag, 0) + 1

        picked, per_artist, seen_titles, known = [], {}, set(), 0
        for i in np.argsort(-s)[: 40*k]:
            artist = self.artist_of[i]
            title = (self.titles[i].lower(), artist)
            if per_artist.get(artist, 0) >= MAX_PER_ARTIST or title in seen_titles:
                continue
            if artist in list_artists:
                if known >= MAX_KNOWN_ARTIST:
                    continue
                known += 1
            per_artist[artist] = per_artist.get(artist, 0) + 1
            seen_titles.add(title)
            picked.append((self.ids[i], self._reason(i, list_artists, list_albums, list_tags)))
            if len(picked) == k:
                break
        return picked

    def _reason(self, i, list_artists, list_albums, list_tags) -> str:
        if self.artist_of[i] in list_artists:
            return "same artist as a song on your playlist"
        if self.album_of[i] in list_albums:
            return "same album as a song on your playlist"
        shared = sorted(self.tags[i] & set(list_tags), key=lambda t: (-list_tags[t], -self.idf[self.vocab[t]]))
        if shared:
            return "similar style: " + ", ".join(shared[:3])
        return "popular song that fits your playlist"

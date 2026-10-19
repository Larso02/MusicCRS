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

            
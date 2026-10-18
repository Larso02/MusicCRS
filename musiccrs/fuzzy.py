"""
Typo tolerant track search

"""
import difflib
import re
import bm25s
import numpy as np

try: 
    from track_database import norm
except ImportError:

    from musiccrs.track_database import norm

def _grams(text: str, n:int=3) -> list[str]:
    s = f" {text} "
    return [s[i:i+n] for i in range(len(s)-n+1)]

class FuzzyTrackIndex:
    def __init__(self, db):
        rows = db.all_norms()
        self.ids = [r["track_id"] for r in rows]
        self.titles = [r["title_norm"] for r in rows]
        self.artists = [r["artist_norm"] for r in rows]
        self.pop = np.array([r["popularity"] for r in rows], dtype=np.float32)
        self.bm25 = bm25s.BM25()
        self.bm25.index([_grams(f"{t} {a}") for t, a in zip(self.titles, self.artists)], show_progress=False,)

    def search(self, text:str, k: int=5, pool: int=40) -> list[tuple[str, float]]:
        variants = {norm(text), norm(re.sub(r"\s+by\s+", " ", text, flags=re.I))}
        variants.discard("")
        if not variants:
            return []

        candidates = set()
        for q in variants:
            scores = self.bm25.get_scores(_grams(q))
            candidates.update(np.argpartition(-scores, pool)[:pool].tolist())
        ranked = []
        for i in candidates:
            forms = (
                self.titles[i],
                f"{self.titles[i]} {self.artists[i]}",
                f"{self.artists[i]} {self.titles[i]}",
            )
            sim = max(
                difflib.SequenceMatcher(None, q, form).ratio()
                for q in variants
                for form in forms 
            )
            ranked.append((sim + 0.02*float(self.pop[i]) / 100, sim, i))
        ranked.sort(reverse=True)
        return [(self.ids[i], sim) for _, sim, i in ranked[:k]]
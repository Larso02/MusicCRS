"""Part 4: feedback-aware ranking.

Three stages on top of the Part 1 BM25 retriever, driven by the session profile (user_profile.py):

1. Rocchio relevance feedback changes the QUERY before retrieval:
       q' = alpha*q + beta*mean(liked tracks) - gamma*mean(disliked tracks)
   q and the track vectors are weighted term vectors over title, artist, album and tag words.
   BM25 retrieves the top K candidates for q' (negative weights are clipped to zero because BM25
   cannot score a negative query weight).
2. A NEURAL RERANKER rescores the K candidates (neural.py): cosine similarity in the embedding spaces
   of the challenge data (Qwen3 metadata embeddings and BPR collaborative-filtering embeddings) to the
   playlist and liked songs, to the user's request, wishes and liked styles embedded as text, minus
   similarity to what the user rejected. BM25 and tag counting play no part in this stage. A few
   non-lexical priors from the Part 1 feature set (same artist or album as the playlist, popularity)
   can be added. Songs and artists the user explicitly rejected are removed as a constraint.
3. Maximal Marginal Relevance (MMR) picks the final list from the best candidates, trading
   relevance against similarity (in the neural space) to the songs already picked.

Every stage can be switched off, which is what the ablation in experiments/10_refine_eval.py does.
Every item keeps its score components, so explanations are computed from facts, not generated.
"""

import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

try:
    from track_database import artist_key, is_descriptive_tag, norm
    from user_profile import Profile
except ImportError:                      # when imported as musiccrs.refine
    from musiccrs.track_database import artist_key, is_descriptive_tag, norm
    from musiccrs.user_profile import Profile

K1, B = 1.5, 0.75                       # BM25 parameters


def tokenize(text: str) -> list[str]:
    return norm(text).split()


class TermIndex:
    """The catalog as weighted term vectors (BM25 weights for scoring, TF-IDF for feedback)."""

    def __init__(self, db_path):
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT track_id, track_name, artist_name, album_name, popularity, tags FROM tracks"
            ).fetchall()
        n = len(rows)
        self.n = n
        self.ids = [r[0] for r in rows]
        self.index = {t: i for i, t in enumerate(self.ids)}
        self.titles = [r[1] for r in rows]
        self.pop = np.array([r[4] or 0 for r in rows], dtype=np.float32) / 100.0

        artist_ids: dict[str, int] = {}
        album_ids: dict[str, int] = {}
        self.artist_of = np.array([artist_ids.setdefault(r[2], len(artist_ids)) for r in rows])
        self.album_of = np.array([album_ids.setdefault(r[3], len(album_ids)) for r in rows])
        self.artist_names: list[list[str]] = [json.loads(r[2]) for r in rows]

        vocab: dict[str, int] = {}
        indptr, indices, counts = [0], [], []
        self.desc_tags: list[frozenset[str]] = []
        artist_docs: dict[str, list[int]] = {}
        tag_docs: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            artists, albums = self.artist_names[i], json.loads(r[3] or "[]")
            raw_tags = json.loads(r[5] or "[]")
            names = {r[1].lower(), *(a.lower() for a in artists), *(a.lower() for a in albums)}
            tags = frozenset(t for t in {x.strip().lower() for x in raw_tags} if is_descriptive_tag(t, names))
            self.desc_tags.append(tags)
            for t in tags:
                tag_docs.setdefault(t, []).append(i)
            for a in artists:
                artist_docs.setdefault(artist_key(a), []).append(i)
            bag = Counter(tokenize(" ".join([r[1], *artists, *albums, *raw_tags])))
            for term, c in bag.items():
                indices.append(vocab.setdefault(term, len(vocab)))
                counts.append(c)
            indptr.append(len(indices))
        self.vocab = vocab
        self.artist_docs = {k: np.array(v) for k, v in artist_docs.items()}
        self.tag_docs = {k: np.array(v) for k, v in tag_docs.items()}

        v = len(vocab)
        tf = sp.csr_matrix((np.array(counts, dtype=np.float32), indices, indptr), shape=(n, v))
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        df = np.bincount(tf.indices, minlength=v)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)

        rows_of = np.repeat(np.arange(n), np.diff(tf.indptr))
        norm_len = K1 * (1 - B + B * doc_len / doc_len.mean())
        w = tf.copy()
        w.data = self.idf[tf.indices] * tf.data * (K1 + 1) / (tf.data + norm_len[rows_of])
        self.bm25_weights = w.tocsr()                                   # N x V

        x = tf.copy()
        x.data = (1 + np.log(tf.data)) * self.idf[tf.indices]
        length = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        length[length == 0] = 1.0
        self.unit = (sp.diags(1.0 / length) @ x).tocsr()                # rows of length 1, for Rocchio and MMR

    # ---------------- vectors ----------------
    def text_vector(self, text: str) -> sp.csr_matrix:
        """Query vector: word counts of the text, summing to 1."""
        bag = Counter(t for t in tokenize(text) if t in self.vocab)
        total = sum(bag.values())
        cols = [self.vocab[t] for t in bag]
        vals = [c / total for c in bag.values()] if total else []
        return sp.csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, len(self.vocab)), dtype=np.float32)

    def doc_vector(self, idx: list[int]) -> sp.csr_matrix:
        """Mean of the tracks' term vectors, summing to 1 (the 'centroid' of a set of tracks)."""
        if not idx:
            return sp.csr_matrix((1, len(self.vocab)), dtype=np.float32)
        mean = sp.csr_matrix(self.unit[idx].mean(axis=0))
        total = mean.sum()
        return (mean / total) if total else mean

    def rocchio(self, q: sp.csr_matrix, liked: list[int], disliked: list[int],
                alpha: float, beta: float, gamma: float, top_terms: int) -> sp.csr_matrix:
        """q' = alpha*q + beta*mean(liked) - gamma*mean(disliked), negatives clipped, top terms kept."""
        new = alpha * q
        if liked:
            new = new + beta * self.doc_vector(liked)
        if disliked:
            new = new - gamma * self.doc_vector(disliked)
        dense = np.asarray(new.todense()).ravel()
        dense[dense < 0] = 0.0
        if top_terms and np.count_nonzero(dense) > top_terms:
            dense[dense < np.partition(dense, -top_terms)[-top_terms]] = 0.0
        return sp.csr_matrix(dense.reshape(1, -1))

    def bm25(self, q: sp.csr_matrix) -> np.ndarray:
        """BM25 score of every track for a weighted query vector."""
        return np.asarray((self.bm25_weights @ q.T).todense()).ravel()

    # ---------------- helpers for the profile ----------------
    def top_tags(self, idx: list[int], k: int = 8) -> list[tuple[str, int]]:
        counts = Counter(t for i in idx for t in self.desc_tags[i])
        return counts.most_common(k)

    def artist_name(self, key: str) -> str:
        """Display name of an artist key (empty if the artist is not in the catalog)."""
        docs = self.artist_docs.get(key)
        if docs is None or len(docs) == 0:
            return ""
        return next((a for a in self.artist_names[int(docs[0])] if artist_key(a) == key), "")

    def label(self, i: int) -> str:
        return f"{self.titles[i]} by {', '.join(self.artist_names[i])}"


@dataclass
class Config:
    # stage 1: Rocchio query modification and BM25 retrieval
    rocchio: bool = True
    alpha: float = 1.0
    beta: float = 0.7
    gamma: float = 0.3
    top_terms: int = 40
    K: int = 1000                       # candidates taken from BM25
    # stage 2: the neural reranker (False = the Part 1 scoring: BM25 + priors, used as the baseline)
    neural: bool = True
    w_learned: float = 1.0               # the trained two-tower network (history -> track)
    w_tracks: float = 0.0               # raw cosine to the playlist and liked songs (metadata space)
    w_cf: float = 0.0                   # ... in the collaborative-filtering space
    w_text: float = 0.5                 # similarity to wishes that match no tag (embedded as text)
    w_styles: float = 0.5               # similarity to the liked styles (mean embedding of the tracks with the tag)
    style_space: str = "learned"        # embedding space of the style prototypes: "learned" (best) or "meta"
    w_negative: float = 1.0             # penalty for similarity to rejected songs ...
    w_style_negative: float = 1.0       # ... and rejected styles
    w_bm25: float = 1.0                 # only for neural=False
    prior_scale: float = 1.0            # multiplies the four priors below (0 = none)
    w_artist_history: float = 1.0
    w_album_history: float = 0.5
    w_last_artist: float = 1.0
    w_pop: float = 1.0
    hard_rejections: bool = True        # remove rejected artists and styles ("no more Coldplay")
    # stage 3: MMR (None = off); lambda in [0, 1]: 1 = relevance only
    mmr_lambda: float | None = None
    mmr_pool: int = 50


@dataclass
class Item:
    idx: int
    score: float
    parts: dict = field(default_factory=dict)       # score components and the facts behind them


def liked_tag_weights(index: TermIndex, profile: Profile, liked: list[int]) -> dict[str, float]:
    """Styles the user likes: stated ones count fully, tags of liked tracks by how common they are
    (used for the explanation text; the neural reranker reads the stated styles as text)."""
    weights = {t: 1.0 for t in profile.liked_tags}
    for tag, count in index.top_tags(liked, 8):
        weights.setdefault(tag, min(1.0, count / max(len(liked), 1)))
    return {t: w for t, w in weights.items() if t not in profile.disliked_tags}


def rejected_tags(index: TermIndex, profile: Profile, liked: list[int], disliked: list[int]) -> set[str]:
    """Styles the user rejects: stated ones, plus tags typical of rejected tracks and not of liked ones."""
    tags = set(profile.disliked_tags)
    keep = {t for i in liked for t in index.desc_tags[i]} | set(profile.liked_tags)
    tags |= {t for t, _ in index.top_tags(disliked, 5) if t not in keep}
    return tags


def refine(index: TermIndex, query_text: str, played: list[int], profile: Profile,
           cfg: Config = Config(), k: int = 20, exclude: list[int] | None = None,
           neural=None, encode=None, request: str = "") -> list[Item]:
    """The top k tracks for the query, given the playlist (played) and the preference profile.

    query_text: text for the BM25 retrieval (for example the whole chat); request: the user's current
    request, embedded for the neural reranker. neural: a neural.NeuralScorer (without it the Part 1
    scoring is used); encode: str -> embedding, for the text anchors. exclude: tracks that must not be
    returned (for example songs already suggested).
    """
    liked = [index.index[t] for t in profile.liked_tracks if t in index.index]
    disliked = [index.index[t] for t in profile.disliked_tracks if t in index.index]

    # stage 1: the query, modified by Rocchio, and retrieval. What the user likes is part of the query
    # (stated artists and styles, wishes), so such tracks are among the candidates and not only reordered.
    liked_names = [index.artist_name(key) for key in sorted(profile.liked_artists)]
    q = index.text_vector(" ".join([query_text, *profile.wishes, *liked_names, *sorted(profile.liked_tags)]))
    if cfg.rocchio and (liked or disliked):
        q = index.rocchio(q, liked, disliked, cfg.alpha, cfg.beta, cfg.gamma, cfg.top_terms)
    elif q.nnz == 0 and played:
        q = index.doc_vector(played)                 # no words to search for: use the playlist itself
    scores = index.bm25(q)
    scores[played + disliked + list(exclude or [])] = -1e9
    K = min(cfg.K, index.n - 1)
    cand = np.argpartition(-scores, K)[:K]
    top = scores[cand].max()
    bm = np.maximum(scores[cand], 0) / top if top > 0 else np.zeros(len(cand))

    # stage 2: reranking
    rej_tags = rejected_tags(index, profile, liked, disliked)
    parts: dict[str, np.ndarray] = {}
    use_neural = cfg.neural and neural is not None
    if use_neural:
        positives = list(dict.fromkeys(liked + played))
        for key in profile.liked_artists:            # a liked artist is represented by its best-known songs
            docs = index.artist_docs.get(key)
            if docs is not None and len(docs):
                positives += [int(d) for d in docs[np.argsort(-index.pop[docs])][:20]]
        text = " ".join([request, *profile.wishes]).strip()
        liked_styles = [index.tag_docs[tag] for tag in sorted(profile.liked_tags) if tag in index.tag_docs]
        rejected_styles = [index.tag_docs[tag] for tag in sorted(rej_tags) if tag in index.tag_docs]
        weights = {"learned": cfg.w_learned, "tracks": cfg.w_tracks, "cf": cfg.w_cf, "text": cfg.w_text,
                   "styles": cfg.w_styles, "disliked_tracks": cfg.w_negative, "disliked_styles": cfg.w_style_negative}
        comps = neural.components(cand, positives, disliked, text, encode, liked_styles, rejected_styles, cfg.style_space)
        for name, z in comps.items():
            parts[name] = weights[name] * z
    else:
        parts["bm25"] = cfg.w_bm25 * bm

    s = cfg.prior_scale                              # priors: no BM25, no tags
    parts["popularity"] = s * cfg.w_pop * index.pop[cand]
    if played:
        parts["artist_history"] = s * cfg.w_artist_history * np.isin(index.artist_of[cand], index.artist_of[played])
        parts["album_history"] = s * cfg.w_album_history * np.isin(index.album_of[cand], index.album_of[played])
        parts["last_artist"] = s * cfg.w_last_artist * (index.artist_of[cand] == index.artist_of[played[-1]])

    tag_w = liked_tag_weights(index, profile, liked)
    total = sum(parts.values())
    if cfg.hard_rejections:                          # a constraint, not a score: "no more Coldplay" is final
        rej = np.zeros(index.n, dtype=bool)
        for tag in rej_tags:
            if tag in index.tag_docs:
                rej[index.tag_docs[tag]] = True
        for key in profile.disliked_artists:
            if key in index.artist_docs:
                rej[index.artist_docs[key]] = True
        total = np.where(rej[cand], total - 1e6, total)
    order = np.argsort(-total)

    # stage 3: MMR over the best candidates
    if cfg.mmr_lambda is not None:
        pool = order[: max(cfg.mmr_pool, k)]
        vectors = None
        if use_neural:
            space = neural.tracks if neural.tracks is not None else neural.meta
            vectors = space[[int(cand[i]) for i in pool]]
        picked = mmr(index, [int(cand[i]) for i in pool], total[pool], k, cfg.mmr_lambda, vectors)
        order = np.array([pool[j] for j in picked])
    order = order[:k]

    items = []
    for i in order:
        d = int(cand[i])
        info = {name: float(v[i]) for name, v in parts.items()}
        info["liked_tags_hit"] = sorted(t for t in index.desc_tags[d] if t in tag_w)
        info["rejected_hit"] = sorted(t for t in index.desc_tags[d] if t in rej_tags)
        info["liked_artist_hit"] = any(artist_key(a) in profile.liked_artists for a in index.artist_names[d])
        items.append(Item(d, float(total[i]), info))
    return items


def mmr(index: TermIndex, docs: list[int], scores: np.ndarray, k: int, lam: float, vectors=None) -> list[int]:
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998). Returns positions in docs.

    vectors: embeddings of docs (neural similarity); without them the term vectors are used.
    """
    if len(docs) == 0:
        return []
    rel = (scores - scores.min()) / (np.ptp(scores) or 1.0)
    if vectors is not None:
        sim = np.asarray(vectors @ vectors.T)
    else:
        sim = (index.unit[docs] @ index.unit[docs].T).toarray()
    chosen: list[int] = []
    while len(chosen) < min(k, len(docs)):
        best, best_value = -1, -1e9
        for j in range(len(docs)):
            if j in chosen:
                continue
            redundancy = max((sim[j, c] for c in chosen), default=0.0)
            value = lam * rel[j] - (1 - lam) * redundancy
            if value > best_value:
                best, best_value = j, value
        chosen.append(best)
    return chosen


def explain(index: TermIndex, item: Item) -> str:
    """Why this track was ranked here, computed from the score components (no LLM involved)."""
    p, reasons = item.parts, []
    if p.get("liked_artist_hit"):
        reasons.append("by an artist you liked")
    if p.get("text", 0) > 0.5:
        reasons.append("matches what you asked for")
    if p.get("styles", 0) > 0.5:
        reasons.append("close to the styles you like")
    if p.get("artist_history", 0) > 0:
        reasons.append("same artist as a song on your playlist")
    elif p.get("album_history", 0) > 0:
        reasons.append("same album as a song on your playlist")
    if p.get("learned", 0) > 0.5 or p.get("tracks", 0) > 0.5:
        reasons.append("fits the songs you chose")
    if p.get("cf", 0) > 0.5:
        reasons.append("listeners of your songs also play it")
    if item.parts.get("liked_tags_hit"):
        reasons.append("styles you like: " + ", ".join(item.parts["liked_tags_hit"][:3]))
    if not reasons:
        reasons.append("close to your request" if p.get("bm25", 0) > 0.5 else "a popular song that fits")
    return "; ".join(reasons)


def diff_lists(before: list[int], after: list[int]) -> dict[str, list[int]]:
    """What the feedback changed: tracks that entered, left and stayed."""
    return {
        "added": [i for i in after if i not in before],
        "removed": [i for i in before if i not in after],
        "kept": [i for i in after if i in before],
    }

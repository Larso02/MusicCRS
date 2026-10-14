"""Feasibility check for the Part 4 proposal (hybrid / embedding-based retrieval).

A) Text-only request (R8-like): query = the current user message only, no history.
   Compare BM25 vs dense retrieval (precomputed Qwen3 embeddings) vs RRF fusion.
B) Playlist-based recommendation (R6-like): turn >= 2, predict the next track from
   the earlier tracks only (no text). Compare the artist/album/popularity heuristic from
   Part 1 with embedding-centroid nearest neighbours (text, CF and audio embeddings).

Uses a fixed random sample of TRAIN turns (seed 0); nothing here touches test.
Needs ~2 GB of downloads on first run (embeddings 790 MB + Qwen3-Embedding-0.6B).
"""
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import bm25s
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from retrieval import BM25Retriever
from retrieval.data_loader import MusicCatalogLoader
from retrieval.turns import load_turns

N_TURNS = 1500
COLS = {
    "attr": "attributes-qwen3_embedding_0.6b",
    "lyr": "lyrics-qwen3_embedding_0.6b",
    "meta": "metadata-qwen3_embedding_0.6b",
    "cf": "cf-bpr",
    "audio": "audio-laion_clap",
}
CACHE = ROOT / "cache" / "emb"
CACHE.mkdir(parents=True, exist_ok=True)

md = MusicCatalogLoader().metadata_dict
bm = BM25Retriever(corpus_types=["track_name", "artist_name", "album_name", "tag_list"])
ids = bm.track_ids
N = len(ids)
tidx = {t: i for i, t in enumerate(ids)}
amap, bmap = {}, {}
artist_of = np.array([amap.setdefault(tuple(md[t]["artist_name"]), len(amap)) for t in ids])
album_of = np.array([bmap.setdefault(tuple(md[t]["album_name"]), len(bmap)) for t in ids])
pop = np.array([md[t]["popularity"] for t in ids], dtype=np.float32) / 100.0


def load_matrices() -> dict[str, np.ndarray]:
    if all((CACHE / f"{k}.npy").exists() for k in COLS):
        return {k: np.load(CACHE / f"{k}.npy") for k in COLS}
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")
    row = {t: i for i, t in enumerate(ds.data.column("track_id").to_pylist())}
    perm = np.array([row[t] for t in ids])
    mats = {}
    for key, col in COLS.items():
        listarr = pa.concat_arrays(ds.data.column(col).chunks)
        lens = pc.fill_null(pc.list_value_length(listarr), 0).to_numpy()
        dim = int(lens.max())
        flat = listarr.flatten().to_numpy(zero_copy_only=False)
        offsets = np.concatenate([[0], np.cumsum(lens)])
        full = np.where(lens == dim)[0]
        arr = np.zeros((len(ds), dim), dtype=np.float32)  # missing embeddings stay all-zero
        arr[full] = flat[offsets[full][:, None] + np.arange(dim)[None, :]]
        print(f"{key}: dim {dim}, tracks without embedding: {int((lens != dim).sum())}", flush=True)
        arr = arr[perm]
        arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
        np.save(CACHE / f"{key}.npy", arr)
        mats[key] = arr
        print("loaded", key, arr.shape, flush=True)
    return mats


def bm_scores(q: str) -> np.ndarray:
    toks = bm25s.tokenize([q.lower()], return_ids=False)[0]
    return bm.bm25_model.get_scores(toks).astype(np.float32)


def ndcg20(scores: np.ndarray, played: np.ndarray, gold: int) -> float:
    s = scores.copy()
    if len(played):
        s[played] = -1e9
    top = np.argpartition(-s, 20)[:20]
    top = top[np.argsort(-s[top])]
    hit = np.where(top == gold)[0]
    return float(1 / np.log2(hit[0] + 2)) if len(hit) else 0.0


def rrf(vecs: list[np.ndarray], played: np.ndarray, weights=None, k=60, depth=200) -> np.ndarray:
    weights = weights or [1.0] * len(vecs)
    out = np.zeros(N, np.float32)
    for v, w in zip(vecs, weights):
        v = v.copy()
        if len(played):
            v[played] = -1e9
        top = np.argpartition(-v, depth)[:depth]
        top = top[np.argsort(-v[top])]
        out[top] += w / (k + np.arange(1, depth + 1))
    return out


def heuristic(hidx: np.ndarray) -> np.ndarray:
    """Part 1 features without any text: artist / album / last-artist / popularity."""
    art = np.isin(artist_of, artist_of[hidx]).astype(np.float32)
    alb = np.isin(album_of, album_of[hidx]).astype(np.float32)
    artl = (artist_of == artist_of[hidx[-1]]).astype(np.float32)
    return 1.0 * art + 0.5 * alb + 1.0 * artl + 1.0 * pop


train = load_turns("train")
random.seed(0)
sample = random.sample(train, N_TURNS)
mats = load_matrices()

print("loading query encoder...", flush=True)
# float32 is essential on CPUs without bfloat16 support (default dtype was ~100x slower here)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"dtype": torch.float32})
texts = [r["user_message"] for r in sample]
t0 = time.time()
qvec = model.encode(texts, prompt_name="query", batch_size=16, normalize_embeddings=True, show_progress_bar=False)
print(f"encoded {len(texts)} queries in {time.time() - t0:.0f}s", flush=True)

lat = []
for text in texts[:10]:
    t = time.time()
    model.encode([text], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)
    lat.append(time.time() - t)
t = time.time()
_ = mats["attr"] @ qvec[0]
print(f"LATENCY single-query embedding (CPU): {1000 * np.mean(lat):.0f} ms; one 47k x 1024 dot product: {1000 * (time.time() - t):.1f} ms", flush=True)

# ---------- A) text-only request ----------
res = {k: [] for k in ["bm25", "dense_attr", "dense_lyr", "dense_meta", "rrf_bm25+attr", "rrf_bm25+meta", "rrf_bm25+attr+meta", "rrf_all4"]}
for r, q in zip(sample, qvec):
    hidx = np.array([tidx[x] for x in r["history_tracks"]], dtype=int)
    gold = tidx[r["gold"]]
    b = bm_scores(r["user_message"])
    d = {k: mats[k] @ q for k in ["attr", "lyr", "meta"]}
    res["bm25"].append(ndcg20(b, hidx, gold))
    for k in d:
        res[f"dense_{k}"].append(ndcg20(d[k], hidx, gold))
    res["rrf_bm25+attr"].append(ndcg20(rrf([b, d["attr"]], hidx), hidx, gold))
    res["rrf_bm25+meta"].append(ndcg20(rrf([b, d["meta"]], hidx), hidx, gold))
    res["rrf_bm25+attr+meta"].append(ndcg20(rrf([b, d["attr"], d["meta"]], hidx), hidx, gold))
    res["rrf_all4"].append(ndcg20(rrf([b, d["attr"], d["meta"], d["lyr"]], hidx), hidx, gold))
print(f"\nA) TEXT-ONLY REQUEST (current message only), n={N_TURNS} train turns, nDCG@20", flush=True)
for k, v in res.items():
    print(f"  {np.mean(v):.4f}  {k}", flush=True)

# ---------- B) playlist-based recommendation ----------
rowsB = [r for r in sample if len(r["history_tracks"]) >= 1]
resB = {k: [] for k in ["heuristic(artist/album/pop)", "centroid_attr", "centroid_meta", "centroid_cf", "centroid_audio",
                       "heur+3*attr", "heur+3*cf", "heur+3*attr+3*cf", "heur+10*attr+10*cf"]}
for r in rowsB:
    hidx = np.array([tidx[x] for x in r["history_tracks"]], dtype=int)
    gold = tidx[r["gold"]]
    h = heuristic(hidx)
    cos = {}
    for k in ["attr", "meta", "cf", "audio"]:
        c = mats[k][hidx].mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-9)
        cos[k] = mats[k] @ c
    resB["heuristic(artist/album/pop)"].append(ndcg20(h, hidx, gold))
    for k in ["attr", "meta", "cf", "audio"]:
        resB[f"centroid_{k}"].append(ndcg20(cos[k], hidx, gold))
    resB["heur+3*attr"].append(ndcg20(h + 3 * cos["attr"], hidx, gold))
    resB["heur+3*cf"].append(ndcg20(h + 3 * cos["cf"], hidx, gold))
    resB["heur+3*attr+3*cf"].append(ndcg20(h + 3 * cos["attr"] + 3 * cos["cf"], hidx, gold))
    resB["heur+10*attr+10*cf"].append(ndcg20(h + 10 * cos["attr"] + 10 * cos["cf"], hidx, gold))
print(f"\nB) PLAYLIST-BASED NEXT-TRACK RECOMMENDATION (turn >= 2, no text), n={len(rowsB)}, nDCG@20", flush=True)
for k, v in resB.items():
    print(f"  {np.mean(v):.4f}  {k}", flush=True)

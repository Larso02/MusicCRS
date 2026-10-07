import sys, os, itertools, time
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np, bm25s, random
from retrieval import BM25Retriever
from retrieval.data_loader import MusicCatalogLoader
from retrieval.turns import load_turns

CT3 = ["track_name", "artist_name", "album_name"]
cat = MusicCatalogLoader(); md = cat.metadata_dict
bm = BM25Retriever(corpus_types=CT3)
ids = bm.track_ids; N = len(ids); tidx = {t: i for i, t in enumerate(ids)}
amap, bmap = {}, {}
artist_of = np.array([amap.setdefault(tuple(md[t]["artist_name"]), len(amap)) for t in ids])
album_of = np.array([bmap.setdefault(tuple(md[t]["album_name"]), len(bmap)) for t in ids])
pop = np.array([md[t]["popularity"] for t in ids], dtype=np.float32) / 100.0

def bm_scores(q):
    toks = bm25s.tokenize([q.lower()], return_ids=False)[0]
    return bm.bm25_model.get_scores(toks).astype(np.float32)

def q_baseline(r):
    lines = []
    for u, tid, a in zip(r["history_user"], r["history_tracks"], r["history_assistant"]):
        lines += [f"user: {u}", f"assistant: {cat.id_to_metadata_str(tid, CT3)}", f"assistant: {a}"]
    lines.append(f"user: {r['user_message']}")
    return "\n".join(lines)

def prep(records):
    out = []
    for r in records:
        sh = bm_scores(q_baseline(r)); sc = bm_scores(r["user_message"])
        sh /= max(sh.max(), 1e-9); sc /= max(sc.max(), 1e-9)
        hidx = np.array([tidx[t] for t in r["history_tracks"]], dtype=int)
        out.append(dict(sh=sh, sc=sc, hidx=hidx, gold=tidx[r["gold"]],
                        art=np.isin(artist_of, artist_of[hidx]) if len(hidx) else np.zeros(N, bool),
                        alb=np.isin(album_of, album_of[hidx]) if len(hidx) else np.zeros(N, bool)))
    return out

def ndcg20(p, w2, a, b, c):
    s = p["sh"] + w2 * p["sc"] + a * p["art"] + b * p["alb"] + c * pop
    if len(p["hidx"]): s[p["hidx"]] = -1e9
    top = np.argpartition(-s, 20)[:20]; top = top[np.argsort(-s[top])]
    hit = np.where(top == p["gold"])[0]
    return 1.0 / np.log2(hit[0] + 2) if len(hit) else 0.0

def run(P, **kw): return float(np.mean([ndcg20(p, **kw) for p in P]))

train = load_turns("train")
random.seed(0); tune_recs = random.sample(train, 1000)
random.seed(1); chk_recs = random.sample(train, 1000)
t0 = time.time(); tuneP = prep(tune_recs); chkP = prep(chk_recs); print("prep done", round(time.time()-t0), "s", flush=True)

print("A  (hist only, filter):        tune", round(run(tuneP, w2=0, a=0, b=0, c=0), 4), " check", round(run(chkP, w2=0, a=0, b=0, c=0), 4), flush=True)
best = None
for w2, a, b, c in itertools.product([0, 0.25, 0.5, 1.0], [0, 0.1, 0.25, 0.5, 1.0], [0, 0.1, 0.25], [0, 0.1, 0.3]):
    v = run(tuneP, w2=w2, a=a, b=b, c=c)
    if best is None or v > best[0]: best = (v, dict(w2=w2, a=a, b=b, c=c))
    if (w2, b, c) == (0, 0, 0) or (w2, a, b, c) in [(0.5, 0.25, 0, 0), (0.5, 0.5, 0.1, 0.1)]:
        print(f"  w2={w2} a={a} b={b} c={c}: tune {v:.4f}", flush=True)
print("BEST on tune:", best, flush=True)
print("BEST on check:", round(run(chkP, **best[1]), 4), flush=True)

import sys, os, itertools, time, random
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np, bm25s
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

test = load_turns("test")
rows = test
T, K = len(rows), 300
SH=np.zeros((T,K),np.float32); SC=SH.copy(); ART=SH.copy(); ALB=SH.copy(); ARTL=SH.copy(); POP=SH.copy()
IDX=np.zeros((T,K),int)
for t, r in enumerate(rows):
    sh = bm_scores(q_baseline(r)); sc = bm_scores(r["user_message"])
    sh /= max(sh.max(),1e-9); sc /= max(sc.max(),1e-9)
    h = np.array([tidx[x] for x in r["history_tracks"]], dtype=int)
    art = np.isin(artist_of, artist_of[h]).astype(np.float32) if len(h) else np.zeros(N,np.float32)
    alb = np.isin(album_of, album_of[h]).astype(np.float32) if len(h) else np.zeros(N,np.float32)
    artl = (artist_of == artist_of[h[-1]]).astype(np.float32) if len(h) else np.zeros(N,np.float32)
    base = sh + 0.25*sc + 2*art + 0.5*alb + 0.5*pop + 0.5*artl
    if len(h): base[h] = -1e9
    top = np.argpartition(-base, K)[:K]
    IDX[t]=top; SH[t]=sh[top]; SC[t]=sc[top]; ART[t]=art[top]; ALB[t]=alb[top]; ARTL[t]=artl[top]; POP[t]=pop[top]
# weights chosen on TRAIN dev set (c.py): w2=0, a=1, b=0.5, c=1.0, d=1
S = SH + 0*SC + 1*ART + 0.5*ALB + 1.0*POP + 1*ARTL
import json
from retrieval.evaluation.evaluate import evaluate
gt = json.load(open("outputs/ground_truth.json"))
def make(lam):
    perm = np.random.RandomState(0).permutation(T)
    usage = np.zeros(N, np.int32); out = [None]*T
    for t in perm:
        s = S[t]
        adj = s - lam*np.log1p(usage[IDX[t]]) if lam>0 else s.copy()
        top = np.argsort(-adj)[:20]
        usage[IDX[t][top]] += 1
        out[t] = [ids[i] for i in IDX[t][top]]
    return out
for lam in [0, 0.25, 0.5, 1.0]:
    lists = make(lam)
    preds = [{"session_id": r["session_id"], "turn_number": r["turn_number"], "predicted_track_ids": l, "predicted_response": ""} for r, l in zip(rows, lists)]
    assert all(len(l)==20 and len(set(l))==20 for l in lists)
    res = evaluate(preds, gt, 47071)
    print(f"TEST lam={lam}: ndcg@20={res['ndcg@20']:.4f} div={res['catalog_diversity']:.4f} FINAL={res['final_score']:.4f}", flush=True)

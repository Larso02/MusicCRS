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

train = load_turns("train")
sess = sorted({r["session_id"] for r in train}); random.seed(0); chosen = set(random.sample(sess, 1000))
rows = [r for r in train if r["session_id"] in chosen]   # 8000 turns, same size as test
T, K = len(rows), 300
SH=np.zeros((T,K),np.float32); SC=SH.copy(); ART=SH.copy(); ALB=SH.copy(); ARTL=SH.copy(); POP=SH.copy()
IDX=np.zeros((T,K),int); GOLD=np.zeros((T,K),bool)
t0=time.time()
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
    GOLD[t] = (top == tidx[r["gold"]])
print("prep", round(time.time()-t0), "s | gold recall within 300 candidates:", round(GOLD.any(1).mean(), 3), flush=True)

def S_of(w2,a,b,c,d): return SH + w2*SC + a*ART + b*ALB + c*POP + d*ARTL
def ndcg_from(S, sel):
    order = np.argsort(-S[sel], axis=1)[:, :20]
    hit = np.take_along_axis(GOLD[sel], order, 1)
    pos = hit.argmax(1); has = hit.any(1)
    return float(np.where(has, 1/np.log2(pos+2), 0).mean())
half = np.arange(T) < T//2; other = ~half
print("A-ish (hist only):  half1", round(ndcg_from(S_of(0,0,0,0,0), half),4), " half2", round(ndcg_from(S_of(0,0,0,0,0), other),4), flush=True)
best=None
for w2,a,b,c,d in itertools.product([0,0.25],[1,2,4],[0.25,0.5,1],[0.3,0.6,1.0],[0,0.5,1]):
    v = ndcg_from(S_of(w2,a,b,c,d), half)
    if best is None or v>best[0]: best=(v,(w2,a,b,c,d))
print("best on half1:", round(best[0],4), dict(zip("w2 a b c d".split(), best[1])), "| half2 check:", round(ndcg_from(S_of(*best[1]), other),4), flush=True)
W = best[1]; S = S_of(*W)
print("full-8000 nDCG@20 with best weights:", round(ndcg_from(S, np.ones(T,bool)),4), flush=True)

def diversify(S, lam, p, seed=0):
    perm = np.random.RandomState(seed).permutation(T)
    usage = np.zeros(N, np.int32); gain = np.zeros(T)
    for t in perm:
        s = S[t]
        adj = s - lam*np.log1p(usage[IDX[t]]) if lam>0 else s.copy()
        if lam>0 and p>0:
            head = np.argpartition(-s, p)[:p]; adj[head] = 1e6 + s[head]
        top = np.argsort(-adj)[:20]
        usage[IDX[t][top]] += 1
        hit = np.where(GOLD[t][top])[0]
        gain[t] = 1/np.log2(hit[0]+2) if len(hit) else 0
    return gain.mean(), (usage>0).sum()/N
print("lam   head  nDCG@20  diversity  final(0.8n+0.2d)", flush=True)
for lam, p in itertools.product([0, 0.1, 0.25, 0.5, 1.0, 2.0], [0, 3, 5, 10]):
    if lam==0 and p>0: continue
    n, dv = diversify(S, lam, p)
    print(f"{lam:<5} {p:<5} {n:.4f}   {dv:.4f}    {0.8*n+0.2*dv:.4f}", flush=True)

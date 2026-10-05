import json
from retrieval import BM25Retriever
from retrieval.turns import load_turns
from retrieval.data_loader import MusicCatalogLoader
import numpy as np, bm25s

CT3 = ["track_name", "artist_name", "album_name"]
catalog = MusicCatalogLoader()
bm25 = BM25Retriever(corpus_types=CT3)
md = catalog.metadata_dict
ids = bm25.track_ids; N = len(ids); tidx = {t:i for i,t in enumerate(ids)}
amap, bmap = {}, {}
artist_of = np.array([amap.setdefault(tuple(md[t]["artist_name"]), len(amap)) for t in ids])
album_of = np.array([bmap.setdefault(tuple(md[t]["album_name"]), len(bmap)) for t in ids])
pop = np.array([md[t]["popularity"] for t in ids], dtype=np.float32) / 100.0

K = 1000
LAM = 1.0
W_ART, W_ALB, W_LAST, W_POP = 1.0, 0.5, 1.0, 1.0

def q_baseline(r):
    lines = []
    for u, tid, a in zip(r["history_user"], r["history_tracks"], r["history_assistant"]):
        lines += [f"user: {u}", f"assistant: {catalog.id_to_metadata_str(tid, CT3)}", f"assistant: {a}"]
    lines.append(f"user: {r['user_message']}")
    return "\n".join(lines)

def bm_scores(q):
    toks = bm25s.tokenize([q.lower()], return_ids=False)[0] 
    return bm25.bm25_model.get_scores(toks).astype(np.float32)

def candidates(r):
    sh = bm_scores(q_baseline(r)); sh /= max(sh.max(), 1e-9)
    h = np.array([tidx[x] for x in r["history_tracks"]], dtype=int)
    art = np.isin(artist_of, artist_of[h]).astype(np.float32) if len(h) else np.zeros(N, np.float32)
    alb = np.isin(album_of, album_of[h]).astype(np.float32) if len(h) else np.zeros(N, np.float32)
    artl = (artist_of == artist_of[h[-1]]).astype(np.float32) if len(h) else np.zeros(N, np.float32)
    score = sh + W_ART*art + W_ALB*alb + W_LAST*artl + W_POP*pop
    if len(h): score[h] = -1e9
    cand = np.argpartition(-score, K)[:K]
    return cand, score[cand]

def diversify(all_cands):
    usage = np.zeros(N, np.float32)
    perm = np.random.RandomState(0).permutation(len(all_cands))
    final = [None] * len(all_cands)
    for t in perm:
        cand, s = all_cands[t]
        adj = s - LAM * np.log1p(usage[cand])
        top = np.argsort(-adj)[:20]
        usage[cand[top]] += 1
        final[t] = [ids[i] for i in cand[top]]
    return final



def main():
    test = load_turns("test")
    all_cands = [candidates(r) for r in test]
    final = diversify(all_cands)
    preds = []
    for r, track_ids in zip(test, final):
        assert len(track_ids) == 20 and len(set(track_ids)) == 20
        preds.append({
            "session_id": r["session_id"],
            "turn_number": r["turn_number"],
            "predicted_track_ids": track_ids,
            "predicted_response": "",
        })
    with open("outputs/predictions.json", "w") as f:
        json.dump(preds, f)
    print("wrote", len(preds))

if __name__ == "__main__":
    main()
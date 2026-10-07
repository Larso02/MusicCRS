import sys, json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np
from collections import Counter
import retrieval.run_final as rf
from retrieval.turns import load_turns
from retrieval.evaluation.evaluate import evaluate
test = load_turns("test"); gt = json.load(open("outputs/ground_truth.json"))
gold = {(g["session_id"], g["turn_number"]): g["ground_truth_track_id"] for g in gt}
rf.K = 1000
all_cands = [rf.candidates(r) for r in test]
def div(lam):
    usage = np.zeros(rf.N, np.float32); perm = np.random.RandomState(0).permutation(len(all_cands)); final = [None]*len(all_cands)
    for t in perm:
        cand, s = all_cands[t]; adj = s - lam*np.log1p(usage[cand]); top = np.argsort(-adj)[:20]
        usage[cand[top]] += 1; final[t] = [rf.ids[i] for i in cand[top]]
    return final
for lam in [0.0, 1.0]:
    final = div(lam)
    preds = [{"session_id": r["session_id"], "turn_number": r["turn_number"], "predicted_track_ids": f, "predicted_response": ""} for r, f in zip(test, final)]
    res = evaluate(preds, gt, 47071)
    print(f"REPO CODE K=1000 lam={lam}: ndcg@1={res['ndcg@1']:.4f} ndcg@10={res['ndcg@10']:.4f} ndcg@20={res['ndcg@20']:.4f} div={res['catalog_diversity']:.4f} final={res['final_score']:.4f}", flush=True)
    if lam == 0.0:
        appear, hits = Counter(), Counter()
        for r, f in zip(test, final):
            g = gold[(r["session_id"], r["turn_number"])]
            for t in f:
                appear[t] += 1
                if t == g: hits[t] += 1
        counts = np.array(list(appear.values()))
        print("lam=0: most-recommended track appears in", counts.max(), "lists; tracks in >=100 lists:", int((counts >= 100).sum()), "; in >=50:", int((counts >= 50).sum()))
        for lo, hi in [(1, 4), (5, 19), (20, 49), (50, 99), (100, 10**9)]:
            ts = [t for t, c in appear.items() if lo <= c <= hi]
            a = sum(appear[t] for t in ts); h = sum(hits[t] for t in ts)
            print(f"  tracks appearing {lo}-{hi if hi < 10**9 else 'inf'} times: {len(ts):5d} tracks, {a:6d} slots, hit rate per slot {h/max(a,1):.5f}", flush=True)

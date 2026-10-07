import sys, json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np
import retrieval.run_final as rf
from retrieval.turns import load_turns
from retrieval.evaluation.evaluate import evaluate
test = load_turns("test"); gt = json.load(open("outputs/ground_truth.json"))
def div(all_cands, lam):
    usage = np.zeros(rf.N, np.float32); perm = np.random.RandomState(0).permutation(len(all_cands)); final = [None]*len(all_cands)
    for t in perm:
        cand, s = all_cands[t]; adj = s - lam*np.log1p(usage[cand]); top = np.argsort(-adj)[:20]
        usage[cand[top]] += 1; final[t] = [rf.ids[i] for i in cand[top]]
    return final
for K in [300, 1000]:
    rf.K = K
    all_cands = [rf.candidates(r) for r in test]
    for lam in [0.5, 1.0, 2.0]:
        final = div(all_cands, lam)
        preds = [{"session_id": r["session_id"], "turn_number": r["turn_number"], "predicted_track_ids": f, "predicted_response": ""} for r, f in zip(test, final)]
        res = evaluate(preds, gt, 47071)
        print(f"K={K} lam={lam}: ndcg@20={res['ndcg@20']:.4f} div={res['catalog_diversity']:.4f} final={res['final_score']:.4f}", flush=True)

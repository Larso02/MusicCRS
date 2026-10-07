import sys, json, time
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import retrieval.run_final as rf
from retrieval.turns import load_turns
from retrieval.evaluation.evaluate import evaluate
test = load_turns("test")
t0 = time.time()
all_cands = [rf.candidates(r) for r in test]
print("stage 1 done in", round(time.time() - t0), "s", flush=True)
final = rf.diversify(all_cands)
preds = [{"session_id": r["session_id"], "turn_number": r["turn_number"],
          "predicted_track_ids": ids20, "predicted_response": ""} for r, ids20 in zip(test, final)]
assert all(len(p["predicted_track_ids"]) == 20 and len(set(p["predicted_track_ids"])) == 20 for p in preds)
gt = json.load(open("outputs/ground_truth.json"))
res = evaluate(preds, gt, 47071)
print("RESULT", {k: round(v, 4) for k, v in res.items()})

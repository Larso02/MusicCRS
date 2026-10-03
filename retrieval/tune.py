import random
from retrieval.evaluation.metrics import compute_ndcg_metrics

def quick_eval(records, retrieve_fn, n=2000, seed=0):
    random.seed(seed)
    sample = random.sample(records, n)
    scores, seen = [], set()
    for r in sample:
        preds = retrieve_fn(r)
        scores.append(compute_ndcg_metrics(preds, [r["gold"]], k_values=[20])["ndcg@20"])
        seen.update(preds)
    ndcg = sum(scores) / len(scores)
    return ndcg, len(seen)
import sys, os, warnings
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from retrieval import BM25Retriever
from retrieval.data_loader import MusicCatalogLoader
from retrieval.turns import load_turns
from retrieval.tune import quick_eval

train = load_turns("train")
catalog = MusicCatalogLoader()
CT3 = ["track_name", "artist_name", "album_name"]
CT4 = CT3 + ["tag_list"]
bm3 = BM25Retriever(corpus_types=CT3)
bm4 = BM25Retriever(corpus_types=CT4)

def q_baseline(r):  # replicates run_bm25_baseline's history string
    lines = []
    for u, tid, a in zip(r["history_user"], r["history_tracks"], r["history_assistant"]):
        lines += [f"user: {u}", f"assistant: {catalog.id_to_metadata_str(tid, CT3)}", f"assistant: {a}"]
    lines.append(f"user: {r['user_message']}")
    return "\n".join(lines)

def q_current(r): return r["user_message"]
def q_cur_hist2(r): return " ".join(r["history_user"][-2:] + [r["user_message"]])
def q_all_user(r): return " ".join(r["history_user"] + [r["user_message"]])
def q_cur_x2_hist(r): return f'{r["user_message"]} {r["user_message"]} ' + " ".join(r["history_user"][-2:])

def fn(query_fn, ret, filt):
    def f(r):
        cand = ret.text_to_item_retrieval(query_fn(r), topk=40 if filt else 20)
        if filt:
            played = set(r["history_tracks"])
            cand = [t for t in cand if t not in played]
        return cand[:20]
    return f

runs = [
 ("BASELINE query, 3 fields, no filter",   q_baseline,    bm3, False),
 ("baseline query, 3 fields, + filter",    q_baseline,    bm3, True),
 ("baseline query, 4 fields(tags), +filter", q_baseline,  bm4, True),
 ("current msg only, 3 fields, +filter",   q_current,     bm3, True),
 ("current msg only, 4 fields, +filter",   q_current,     bm4, True),
 ("cur + last2 user msgs, 4 fields",       q_cur_hist2,   bm4, True),
 ("all user msgs, 4 fields",               q_all_user,    bm4, True),
 ("cur x2 + last2 user, 4 fields",         q_cur_x2_hist, bm4, True),
]
for name, qf, ret, filt in runs:
    nd, _ = quick_eval(train, fn(qf, ret, filt), n=2000, seed=0)
    print(f"{nd:.4f}  {name}", flush=True)

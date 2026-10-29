"""Part 4 evaluation: Rocchio + feedback reranker + MMR against the Part 1 baseline (train sessions only).

A) Implicit feedback, ablation. For turns with at least one earlier track, the earlier tracks are the
   playlist and also the liked tracks. The query is the chat text (Part 1 style). nDCG@20 against the
   gold track of the turn for: Part 1 features only, + Rocchio, + reranker, both.
B) Explicit negative feedback. For turns where the Part 1 top-5 is dominated by one tag (>= 3 of 5)
   that the gold track does NOT have, the user rejects that tag. We measure the share of top-5 results
   with the rejected tag before/after and nDCG@20 before/after (same for rejecting an artist).
C) Diversity. Intra-list diversity of the top-5 with and without MMR.

Usage (repo root):  python experiments/10_refine_eval.py [--turns 1500] [--tune]
Nothing here touches the test split.
"""
import argparse
import itertools
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from retrieval.turns import load_turns                      # noqa: E402
from musiccrs.refine import Config, TermIndex, refine        # noqa: E402
from musiccrs.track_database import DATABASE_PATH            # noqa: E402
from musiccrs.user_profile import Profile                    # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--turns", type=int, default=1500)
parser.add_argument("--scenarios", type=int, default=400)
parser.add_argument("--tune", action="store_true", help="grid-search the weights on the first half")
args = parser.parse_args()

t0 = time.time()
ix = TermIndex(DATABASE_PATH)
print(f"index: {ix.n} tracks, {len(ix.vocab)} terms, built in {time.time() - t0:.0f} s", flush=True)

random.seed(0)
turns = [r for r in load_turns("train") if r["history_tracks"] and r["gold"] in ix.index]
sample = random.sample(turns, args.turns)
print(f"{len(sample)} train turns with history", flush=True)


def query_text(r):
    lines = []
    for u, tid, a in zip(r["history_user"], r["history_tracks"], r["history_assistant"]):
        lines += [u, ix.label(ix.index[tid]) if tid in ix.index else "", a]
    lines.append(r["user_message"])
    return "\n".join(lines)


def ndcg(items, gold_idx, k=20):
    for rank, it in enumerate(items[:k]):
        if it.idx == gold_idx:
            return 1.0 / np.log2(rank + 2)
    return 0.0


def implicit_profile(r):
    p = Profile()
    for tid in r["history_tracks"]:
        if tid in ix.index:
            p.like_track(tid)
    return p


def run_a(cfg, rows):
    scores = []
    for r in rows:
        played = [ix.index[t] for t in r["history_tracks"] if t in ix.index]
        items = refine(ix, query_text(r), played, implicit_profile(r), cfg, k=20)
        scores.append(ndcg(items, ix.index[r["gold"]]))
    return float(np.mean(scores))


BASE = Config(rocchio=False, rerank=False)
half = len(sample) // 2
A, Bh = sample[:half], sample[half:]

if args.tune:
    print("\ntuning on the first half ...", flush=True)
    best = (run_a(Config(), A), Config())
    for beta, gamma, w_tag in itertools.product([0.4, 0.7, 1.2], [0.0, 0.3], [0.5, 1.0, 2.0]):
        cfg = Config(beta=beta, gamma=gamma, w_tag=w_tag)
        s = run_a(cfg, A)
        if s > best[0]:
            best = (s, cfg)
            print(f"  new best {s:.4f}: beta={beta} gamma={gamma} w_tag={w_tag}", flush=True)
    TUNED = best[1]
    print(f"chosen: beta={TUNED.beta} gamma={TUNED.gamma} w_tag={TUNED.w_tag}", flush=True)
else:
    TUNED = Config()

print("\nA) implicit feedback, nDCG@20 (first half / second half):")
variants = {
    "Part 1 features only": BASE,
    "+ Rocchio": replace(TUNED, rerank=False),
    "+ reranker (tags, liked artists)": replace(TUNED, rocchio=False),
    "+ Rocchio + reranker": TUNED,
}
for name, cfg in variants.items():
    a, b = run_a(cfg, A), run_a(cfg, Bh)
    print(f"  {np.mean([a, b]):.4f}  ({a:.4f} / {b:.4f})  {name}", flush=True)

# ---------- B) explicit negative feedback ----------
print("\nB) rejecting a tag / an artist the gold track does not have")
rows_tag, rows_art = [], []
for r in sample[: args.scenarios * 3]:
    played = [ix.index[t] for t in r["history_tracks"] if t in ix.index]
    base = refine(ix, query_text(r), played, implicit_profile(r), BASE, k=20)
    gold = ix.index[r["gold"]]
    top5 = base[:5]
    counts = {}
    for it in top5:
        for t in ix.desc_tags[it.idx]:
            counts[t] = counts.get(t, 0) + 1
    cands = [t for t, c in counts.items() if c >= 3 and t not in ix.desc_tags[gold]]
    if cands:
        rows_tag.append((r, played, base, max(cands, key=lambda t: counts[t])))
    keys = {}
    for it in top5:
        for a in ix.artist_names[it.idx]:
            keys[a] = keys.get(a, 0) + 1
    arts = [a for a, c in keys.items() if c >= 2 and a not in ix.artist_names[gold]]
    if arts:
        rows_art.append((r, played, base, max(arts, key=lambda a: keys[a])))
    if len(rows_tag) >= args.scenarios and len(rows_art) >= args.scenarios:
        break
print(f"  {len(rows_tag)} tag scenarios, {len(rows_art)} artist scenarios")


def share(items, test):
    return float(np.mean([test(it.idx) for it in items[:5]]))


def run_b(cfg, rows, kind):
    before, after, nd_b, nd_a = [], [], [], []
    for r, played, base, target in rows:
        p = implicit_profile(r)
        if kind == "tag":
            p.dislike_tag(target)
            test = lambda i, t=target: t in ix.desc_tags[i]
        else:
            from musiccrs.track_database import artist_key
            p.dislike_artist(artist_key(target))
            test = lambda i, t=target: t in ix.artist_names[i]
        new = refine(ix, query_text(r), played, p, cfg, k=20)
        gold = ix.index[r["gold"]]
        before.append(share(base, test)); after.append(share(new, test))
        nd_b.append(ndcg(base, gold)); nd_a.append(ndcg(new, gold))
    b, a = np.mean(before), np.mean(after)
    return b, a, (1 - a / b) if b else 0.0, np.mean(nd_b), np.mean(nd_a)


for kind, rows in (("tag", rows_tag), ("artist", rows_art)):
    print(f"  rejected {kind}: share of top-5 before -> after, reduction, nDCG@20 before -> after")
    for name, cfg in {"+ Rocchio": replace(TUNED, rerank=False), "+ reranker": replace(TUNED, rocchio=False),
                      "+ Rocchio + reranker": TUNED}.items():
        b, a, red, nb, na = run_b(cfg, rows, kind)
        print(f"    {name:22} {b:.2f} -> {a:.2f}  cut {100 * red:4.0f}%   nDCG {nb:.4f} -> {na:.4f}", flush=True)

# ---------- C) MMR ----------
print("\nC) diversity of the top-5 (1 - mean pairwise cosine of the term vectors) and nDCG@20")
for lam in (None, 0.8, 0.6, 0.4):
    div, nd = [], []
    for r in sample[:500]:
        played = [ix.index[t] for t in r["history_tracks"] if t in ix.index]
        items = refine(ix, query_text(r), played, implicit_profile(r), replace(TUNED, mmr_lambda=lam), k=20)
        sims = (ix.unit[[i.idx for i in items[:5]]] @ ix.unit[[i.idx for i in items[:5]]].T).toarray()
        div.append(1 - (sims.sum() - 5) / 20)
        nd.append(ndcg(items, ix.index[r["gold"]]))
    print(f"  lambda={lam}: diversity {np.mean(div):.3f}, nDCG@20 {np.mean(nd):.4f}", flush=True)

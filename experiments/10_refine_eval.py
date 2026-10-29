"""Part 4 evaluation: Rocchio + NEURAL reranker + MMR against the Part 1 baseline (train sessions only).

A) Implicit feedback, ablation. For turns with at least one earlier track (in held-out sessions), the
   earlier tracks are the playlist and also the liked tracks. BM25 retrieves with the chat text (Part 1 style); the neural
   reranker uses the trained network on the earlier tracks. nDCG@20 against the gold track of the turn for:
   Part 1 scoring (BM25 + priors), + Rocchio, + neural reranker, both. Then ablations of the neural parts.
B) Explicit negative feedback. For turns where the Part 1 top-5 is dominated by one tag (>= 3 of 5)
   that the gold track does NOT have, the user rejects that tag. We measure the share of top-5 results
   with the rejected tag before/after and nDCG@20 before/after (same for rejecting an artist).
C) Diversity. Intra-list diversity of the top-5 with and without MMR (neural similarity).

Usage (repo root):  python experiments/10_refine_eval.py [--turns 1500] [--tune]
Nothing here touches the test split. Query embeddings are cached in cache/emb/eval_queries.npz.
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
from musiccrs.neural import NeuralScorer                    # noqa: E402
from musiccrs.playlist_generator import QUERY_MODEL         # noqa: E402
from musiccrs.refine import Config, TermIndex, refine        # noqa: E402
from musiccrs.reranker_model import is_heldout               # noqa: E402
from musiccrs.track_database import DATABASE_PATH, artist_key  # noqa: E402
from musiccrs.user_profile import Profile                    # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--turns", type=int, default=1500)
parser.add_argument("--scenarios", type=int, default=400)
parser.add_argument("--tune", action="store_true", help="grid-search the weights on the first half")
args = parser.parse_args()

t0 = time.time()
ix = TermIndex(DATABASE_PATH)
neural = NeuralScorer.load(ix.ids)
print(f"index: {ix.n} tracks, {len(ix.vocab)} terms; embeddings loaded; {time.time() - t0:.0f} s", flush=True)

random.seed(0)
turns = [r for r in load_turns("train") if r["history_tracks"] and r["gold"] in ix.index
         and is_heldout(r["session_id"])]       # sessions the neural reranker was NOT trained on
sample = random.sample(turns, args.turns)
print(f"{len(sample)} held-out train turns with history (of {len(turns)})", flush=True)


class CachedEncoder:
    """Qwen3-Embedding query encoder (the model that made the track embeddings) with a disk cache."""

    path = ROOT / "cache" / "emb" / "eval_queries.npz"

    def __init__(self):
        self.vectors: dict[str, np.ndarray] = {}
        if self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            self.vectors = dict(zip(data["texts"].tolist(), data["vectors"]))
        self.model = None

    def _model(self):
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(QUERY_MODEL, model_kwargs={"dtype": torch.float32})
        return self.model

    def warm(self, texts):
        todo = sorted({t for t in texts if t not in self.vectors})
        if not todo:
            return
        print(f"encoding {len(todo)} texts ...", flush=True)
        start = time.time()
        out = self._model().encode(todo, prompt_name="query", batch_size=16, normalize_embeddings=True,
                                   show_progress_bar=False)
        self.vectors.update(zip(todo, out))
        np.savez(self.path, texts=np.array(list(self.vectors)), vectors=np.stack(list(self.vectors.values())))
        print(f"  done in {time.time() - start:.0f} s", flush=True)

    def __call__(self, text):
        if text not in self.vectors:
            self.warm([text])
        return self.vectors[text]


encode = CachedEncoder()                      # only styles and wishes are embedded, as in the chat


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


def rank(r, cfg, profile=None, k=20, text=None):
    """text=None: BM25 retrieval with the whole chat (Part 1 style); text="": as in the chat system,
    where only the profile (wishes, liked styles, liked songs) forms the retrieval query."""
    played = [ix.index[t] for t in r["history_tracks"] if t in ix.index]
    return refine(ix, query_text(r) if text is None else text, played, profile or implicit_profile(r), cfg, k=k,
                  neural=neural, encode=encode)


def run_a(cfg, rows):
    return float(np.mean([ndcg(rank(r, cfg), ix.index[r["gold"]]) for r in rows]))


BASE = Config(rocchio=False, neural=False)                   # Part 1 scoring: BM25 + priors
half = len(sample) // 2
A, Bh = sample[:half], sample[half:]

if args.tune:
    print("\ntuning on the first 400 turns of the first half ...", flush=True)
    tune_rows = A[:400]
    best = Config()
    best_score = run_a(best, tune_rows)
    print(f"  start {best_score:.4f}", flush=True)
    for beta in (0.4, 0.7, 1.2):                             # Rocchio
        cfg = replace(best, beta=beta)
        s = run_a(cfg, tune_rows)
        if s > best_score:
            best, best_score = cfg, s
            print(f"  beta={beta}: {s:.4f}", flush=True)
    for w_tracks, w_cf, scale, w_neg in itertools.product((0.0, 0.5), (0.0, 0.5), (0.0, 0.5, 1.0), (1.0,)):
        cfg = replace(best, w_tracks=w_tracks, w_cf=w_cf, prior_scale=scale, w_negative=w_neg)
        s = run_a(cfg, tune_rows)
        if s > best_score:
            best, best_score = cfg, s
            print(f"  w_tracks={w_tracks} w_cf={w_cf} prior_scale={scale}: {s:.4f}", flush=True)
    TUNED = best
    print(f"chosen: beta={TUNED.beta} w_learned={TUNED.w_learned} w_tracks={TUNED.w_tracks} w_cf={TUNED.w_cf} "
          f"prior_scale={TUNED.prior_scale}", flush=True)
else:
    TUNED = Config()

print("\nA) implicit feedback, nDCG@20 (first half / second half):")
variants = {
    "Part 1 scoring (BM25 + priors)": BASE,
    "+ Rocchio": replace(TUNED, neural=False),
    "+ neural reranker": replace(TUNED, rocchio=False),
    "+ Rocchio + neural reranker": TUNED,
    "   ablation: neural without priors": replace(TUNED, prior_scale=0.0),
    "   ablation: trained network only": replace(TUNED, w_tracks=0.0, w_cf=0.0, prior_scale=0.0),
    "   ablation: zero-shot cosines only (no trained network)": replace(TUNED, w_learned=0.0, w_tracks=1.0, w_cf=0.5),
}
for name, cfg in variants.items():
    a, b = run_a(cfg, A), run_a(cfg, Bh)
    print(f"  {np.mean([a, b]):.4f}  ({a:.4f} / {b:.4f})  {name}", flush=True)

# ---------- B) explicit negative feedback ----------
print("\nB) rejecting a tag / an artist the gold track does not have")
rows_tag, rows_art = [], []
for r in sample[: args.scenarios * 3]:
    base = rank(r, BASE)
    gold = ix.index[r["gold"]]
    top5 = base[:5]
    counts = {}
    for it in top5:
        for t in ix.desc_tags[it.idx]:
            counts[t] = counts.get(t, 0) + 1
    cands = [t for t, c in counts.items() if c >= 3 and t not in ix.desc_tags[gold]]
    if cands:
        rows_tag.append((r, base, max(cands, key=lambda t: counts[t])))
    keys = {}
    for it in top5:
        for a in ix.artist_names[it.idx]:
            keys[a] = keys.get(a, 0) + 1
    arts = [a for a, c in keys.items() if c >= 2 and a not in ix.artist_names[gold]]
    if arts:
        rows_art.append((r, base, max(arts, key=lambda a: keys[a])))
    if len(rows_tag) >= args.scenarios and len(rows_art) >= args.scenarios:
        break
print(f"  {len(rows_tag)} tag scenarios, {len(rows_art)} artist scenarios")


def share(items, test):
    return float(np.mean([test(it.idx) for it in items[:5]]))


def run_b(cfg, rows, kind):
    before, after, nd_b, nd_a = [], [], [], []
    for r, base, target in rows:
        p = implicit_profile(r)
        if kind == "tag":
            p.dislike_tag(target)
            test = lambda i, t=target: t in ix.desc_tags[i]
        else:
            p.dislike_artist(artist_key(target))
            test = lambda i, t=target: t in ix.artist_names[i]
        new = rank(r, cfg, p)
        gold = ix.index[r["gold"]]
        before.append(share(base, test)); after.append(share(new, test))
        nd_b.append(ndcg(base, gold)); nd_a.append(ndcg(new, gold))
    b, a = np.mean(before), np.mean(after)
    return b, a, (1 - a / b) if b else 0.0, np.mean(nd_b), np.mean(nd_a)


for kind, rows in (("tag", rows_tag), ("artist", rows_art)):
    print(f"  rejected {kind}: share of top-5 before -> after, reduction, nDCG@20 before -> after")
    for name, cfg in {"+ Rocchio only": replace(TUNED, neural=False, hard_rejections=False),
                      "+ neural reranker, no constraint": replace(TUNED, rocchio=False, hard_rejections=False),
                      "+ Rocchio + neural, no constraint": replace(TUNED, hard_rejections=False),
                      "+ Rocchio + neural, penalty x3, no constr.": replace(TUNED, hard_rejections=False, w_negative=3.0, w_style_negative=3.0),
                      "+ Rocchio + neural + constraint": TUNED}.items():
        b, a, red, nb, na = run_b(cfg, rows, kind)
        print(f"    {name:36} {b:.2f} -> {a:.2f}  cut {100 * red:4.0f}%   nDCG {nb:.4f} -> {na:.4f}", flush=True)


# ---------- D) "more of style X": the text anchor ----------
print("\nD) the user asks for more of a style that the gold track has and the history does not")
rows_like = []
for r in sample[: args.scenarios * 2]:
    gold = ix.index[r["gold"]]
    seen = {t for tid in r["history_tracks"] if tid in ix.index for t in ix.desc_tags[ix.index[tid]]}
    cands = [t for t in ix.desc_tags[gold] if t not in seen and len(ix.tag_docs.get(t, [])) >= 50]
    if cands:
        rows_like.append((r, min(cands, key=lambda t: len(ix.tag_docs[t]))))        # the most specific one
    if len(rows_like) >= args.scenarios:
        break
print(f"  {len(rows_like)} scenarios; share of top-5 with the style, nDCG@20")


def run_d(cfg, use_tag):
    shares, scores = [], []
    for r, tag in rows_like:
        p = implicit_profile(r)
        if use_tag:
            p.like_tag(tag)
        items = rank(r, cfg, p, text="")
        shares.append(share(items, lambda i, t=tag: t in ix.desc_tags[i]))
        scores.append(ndcg(items, ix.index[r["gold"]]))
    return np.mean(shares), np.mean(scores)


for name, cfg, use in (("style not mentioned", TUNED, False),
                       ("style in the query only (no neural style)", replace(TUNED, w_styles=0.0), True),
                       ("style prototype, metadata space", replace(TUNED, style_space="meta"), True),
                       ("style prototype, trained space", replace(TUNED, style_space="learned"), True),
                       ("style prototype, trained space, weight 0.5", replace(TUNED, style_space="learned", w_styles=0.5), True),
                       ("style prototype, trained space, weight 0.25", replace(TUNED, style_space="learned", w_styles=0.25), True),
                       ("style prototype, trained space, weight 2", replace(TUNED, style_space="learned", w_styles=2.0), True)):
    s, n = run_d(cfg, use)
    print(f"    {name:46} share {s:.2f}   nDCG@20 {n:.4f}", flush=True)

# ---------- C) MMR ----------
print("\nC) diversity of the top-5 (1 - mean pairwise cosine, neural space) and nDCG@20")
for lam in (None, 0.8, 0.6, 0.4):
    div, nd = [], []
    for r in sample[:500]:
        items = rank(r, replace(TUNED, mmr_lambda=lam))
        m = neural.meta[[i.idx for i in items[:5]]]
        sims = m @ m.T
        div.append(1 - (sims.sum() - 5) / 20)
        nd.append(ndcg(items, ix.index[r["gold"]]))
    print(f"  lambda={lam}: diversity {np.mean(div):.3f}, nDCG@20 {np.mean(nd):.4f}", flush=True)

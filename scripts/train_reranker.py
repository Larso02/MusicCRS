"""Train the neural reranker (two-tower network) on the TalkPlay training sessions.

For every training turn with at least one earlier track, the model sees the earlier tracks and must
rank the track that was actually played next above other tracks (sampled softmax over the gold track,
the other golds in the batch and random catalog tracks drawn by popularity). Every 10th session is
held out and never used for training; experiments/10_refine_eval.py evaluates on those sessions.

Usage, from the repo root (needs cache/emb/meta.npy and cf.npy; they are built on first use):
    python -m scripts.train_reranker                 # about 10-15 minutes on a laptop CPU
    python -m scripts.train_reranker --epochs 3

Writes data/reranker.pt (about 3 MB).
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from musiccrs.neural import load_embeddings            # noqa: E402
from musiccrs.reranker_model import MAX_HISTORY, TwoTower, is_heldout, save  # noqa: E402
from musiccrs.track_database import DATABASE_PATH      # noqa: E402
from retrieval.turns import load_turns                 # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--negatives", type=int, default=2048)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

import sqlite3                                          # noqa: E402
with sqlite3.connect(DATABASE_PATH) as conn:
    rows = conn.execute("SELECT track_id, popularity FROM tracks").fetchall()
ids = [r[0] for r in rows]
index = {t: i for i, t in enumerate(ids)}
pop = np.array([r[1] or 0 for r in rows], dtype=np.float64) + 1.0
neg_probs = torch.tensor(pop ** 0.75 / (pop ** 0.75).sum(), dtype=torch.float)

meta, cf = load_embeddings(ids, "meta"), load_embeddings(ids, "cf")
features = torch.tensor(np.concatenate([meta, cf], axis=1))          # (N, 1152)
print(f"features {tuple(features.shape)}", flush=True)


def build(turns):
    hist = np.full((len(turns), MAX_HISTORY), -1, dtype=np.int64)
    gold = np.zeros(len(turns), dtype=np.int64)
    for n, r in enumerate(turns):
        h = [index[t] for t in r["history_tracks"] if t in index][-MAX_HISTORY:]
        hist[n, MAX_HISTORY - len(h):] = h
        gold[n] = index[r["gold"]]
    return torch.tensor(hist), torch.tensor(gold)


all_turns = [r for r in load_turns("train") if r["history_tracks"] and r["gold"] in index]
train_turns = [r for r in all_turns if not is_heldout(r["session_id"])]
val_turns = [r for r in all_turns if is_heldout(r["session_id"])][:3000]
print(f"{len(train_turns)} training turns, {len(val_turns)} held-out turns for monitoring", flush=True)
train_hist, train_gold = build(train_turns)
val_hist, val_gold = build(val_turns)

model = TwoTower(features.shape[1])
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
steps = args.epochs * (len(train_turns) // args.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)


def user_vectors(hist, tower_inputs=None):
    mask = hist >= 0
    safe = hist.clamp(min=0)
    enc = model.encode_tracks(features[safe.reshape(-1)]).reshape(*hist.shape, -1)
    return model.encode_user(enc, mask)


@torch.no_grad()
def evaluate():
    """Recall@20 and nDCG@20 over the whole catalog on held-out turns (history excluded)."""
    model.eval()
    catalog = model.encode_tracks(features)
    hits, ndcg = 0.0, 0.0
    for start in range(0, len(val_hist), 500):
        h, g = val_hist[start:start + 500], val_gold[start:start + 500]
        scores = user_vectors(h) @ catalog.T
        for row in range(len(h)):
            scores[row, h[row][h[row] >= 0]] = -9
        top = scores.topk(20, dim=1).indices
        found = (top == g[:, None])
        hits += found.any(1).float().sum().item()
        pos = found.float().argmax(1)
        ndcg += (found.any(1).float() / torch.log2(pos.float() + 2)).sum().item()
    model.train()
    return hits / len(val_hist), ndcg / len(val_hist)


print("before training: recall@20 %.4f, nDCG@20 %.4f" % evaluate(), flush=True)
step, start = 0, time.time()
for epoch in range(args.epochs):
    order = torch.randperm(len(train_turns))
    total = 0.0
    for b in range(len(train_turns) // args.batch):
        sel = order[b * args.batch:(b + 1) * args.batch]
        hist, gold = train_hist[sel], train_gold[sel]
        negatives = torch.multinomial(neg_probs, args.negatives, replacement=True)
        cands = torch.cat([gold, negatives])                       # gold of the batch + shared random negatives
        user = user_vectors(hist)
        tracks = model.encode_tracks(features[cands])
        logits = model.scale() * (user @ tracks.T)
        # remove accidental positives: a candidate equal to this row's gold (other than its own slot)
        same = cands[None, :] == gold[:, None]
        same[torch.arange(len(gold)), torch.arange(len(gold))] = False
        logits = logits.masked_fill(same, -1e9)
        loss = F.cross_entropy(logits, torch.arange(len(gold)))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        total += loss.item(); step += 1
    recall, ndcg = evaluate()
    print(f"epoch {epoch + 1}/{args.epochs}: loss {total / (len(train_turns) // args.batch):.3f}, held-out recall@20 {recall:.4f}, "
          f"nDCG@20 {ndcg:.4f}  ({time.time() - start:.0f} s)", flush=True)

model.eval()
save(model)
print("saved data/reranker.pt")

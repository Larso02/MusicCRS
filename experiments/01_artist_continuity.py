import sys, os
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from collections import Counter
from retrieval.data_loader import MusicCatalogLoader
from retrieval.turns import load_turns
cat = MusicCatalogLoader().metadata_dict
train = load_turns("train")
art = lambda t: tuple(cat[t]["artist_name"])
alb = lambda t: tuple(cat[t]["album_name"])
by_artist_size = Counter(art(t) for t in cat)
def stats(rows):
    n = len(rows)
    a = sum(art(r["gold"]) in {art(h) for h in r["history_tracks"]} for r in rows) / n
    b = sum(alb(r["gold"]) in {alb(h) for h in r["history_tracks"]} for r in rows) / n
    return n, a, b
print("turn  n       gold artist in history   gold album in history")
for t in range(2, 9):
    n, a, b = stats([r for r in train if r["turn_number"] == t])
    print(f"{t}     {n:6d}  {a:8.3f}                 {b:8.3f}")
rows = [r for r in train if r["turn_number"] >= 2]
n, a, b = stats(rows); print(f"all t>=2 {n}  {a:.3f}  {b:.3f}")
# how many candidate tracks would "same artist as last played track" give?
sizes = [by_artist_size[art(r["history_tracks"][-1])] - 1 for r in rows]
import statistics
print("tracks by same artist as last played: median", statistics.median(sizes), "mean", round(sum(sizes)/len(sizes),1))

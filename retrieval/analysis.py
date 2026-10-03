from retrieval.turns import load_turns
from retrieval.data_loader import MusicCatalogLoader

train = load_turns("train")
print("turns:", len(train))

repeat = sum(r["gold"] in r["history_tracks"] for r in train) / len(train)
print("gold already in history:", repeat)

golds = [r["gold"] for r in train]
print("unique golds:", len(set(golds)), "of 47071")

cat = MusicCatalogLoader().metadata_dict
pop = lambda t: float(cat[t]["popularity"])
print("Mean gold pop:", sum(map(pop, golds)) / len(golds), "catalog mean:", sum(map(pop, cat)) / len(cat))

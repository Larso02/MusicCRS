from retrieval import BM25Retriever
from retrieval.turns import load_turns
from retrieval.tune import quick_eval

train = load_turns("train")
bm25 = BM25Retriever(corpus_types= ["track_name", "artist_name", "album_name", "tag_list"])


def q_current(r):
    return r["user_message"]

def q_weighted(r):
    recent = " ".join(r["history_user"][-2])
    return f'{r["user_message"]} {r["user_message"]} {recent}'

def make_fn(query_fn, retriever=bm25):
    def fn(r):
        played = set(r["history_tracks"])
        cand = retriever.text_to_item_retrieval(query_fn(r), topk=40)
        return [t for t in cand if t not in played][:20]
    return fn

print("current only:", quick_eval(train, make_fn(q_current)))
import json, warnings, os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
from datasets import load_dataset
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
print(ds.column_names)
s = ds[0]
for k,v in s.items():
    if k != "conversations": print(k, "=>", str(v)[:300])
for m in s["conversations"][:9]:
    print({k:(str(v)[:160]) for k,v in m.items()})
meta = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
print(meta.column_names); print({k:str(v)[:200] for k,v in meta[0].items()})
emb = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks", streaming=True)
e = next(iter(emb)); print({k:(len(v) if hasattr(v,'__len__') and not isinstance(v,str) else v) for k,v in e.items()})

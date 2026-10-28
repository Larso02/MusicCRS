"""Label the catalog's tags with the LLM: does a tag describe the music, or is it noise?

The tags are free text typed by Last.fm users ("rock", "mellow", but also "favorites",
"albums i own", "2013", "seen live"). Instead of a hand-written stop list, the LLM judges each
tag once, offline, and the answer is stored in data/tag_labels.json. The agent reads that file;
it makes no LLM call at run time.

Only tags used by at least MIN_TRACKS tracks are judged; rarer tags are typos or private
labels and are dropped by the frequency rule alone.

Usage, from the repo root (needs MY_OLLAMA_API_KEY in .env; the run can be stopped and resumed):
    python -m scripts.label_tags                 # judge everything not judged yet
    python -m scripts.label_tags --limit 300     # try the 300 most common tags first
"""

import argparse
import json
import os
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "tracks.db"
LABELS_PATH = ROOT / "data" / "tag_labels.json"

LLM_BASE_URL = "https://openwebui.ux.uis.no/api"
LLM_MODEL = "gorina10.gemma-4-31b"
MIN_TRACKS = 10          # tags used by fewer tracks are not judged (and not used)
BATCH = 120

PROMPT = """You are cleaning the tags of a music catalog. The tags were typed freely by listeners.

Below is a numbered list of tags. Return the numbers of the tags that DESCRIBE THE MUSIC ITSELF:
genre or sub-genre, mood or emotion, instruments, vocals, tempo or energy, era or decade,
language, country or region of origin, or a situation the music suits (for example workout, study).

Do NOT keep tags that are personal opinions or ratings, ownership or listening history
(favorites, seen live, albums I own), single years, names of artists, bands, songs or albums,
meaningless words, or typos.

Answer with ONLY a JSON array of the numbers to keep, for example [1, 4, 7].

Tags:
{tags}"""


def tag_counts() -> Counter:
    counts: Counter = Counter()
    with sqlite3.connect(DATABASE_PATH) as conn:
        for (tags,) in conn.execute("SELECT tags FROM tracks"):
            counts.update({t.strip().lower() for t in json.loads(tags or "[]")} - {""})
    return counts


def judge(client: OpenAI, tags: list[str]) -> set[str] | None:
    """The subset of tags the LLM keeps, or None if the answer could not be read."""
    numbered = "\n".join(f"{i}. {tag}" for i, tag in enumerate(tags, start=1))
    for _ in range(3):
        try:
            reply = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0,
                messages=[{"role": "user", "content": PROMPT.format(tags=numbered)}],
            ).choices[0].message.content or ""
            match = re.search(r"\[[\d,\s]*\]", reply)
            if match:
                keep = {int(n) for n in json.loads(match.group(0))}
                return {tags[n - 1] for n in keep if 1 <= n <= len(tags)}
        except Exception as error:      # network hiccup, timeout, rate limit: try again
            print("  retry:", type(error).__name__)
            time.sleep(2)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="only the N most common tags (a trial run)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.getenv("MY_OLLAMA_API_KEY"), base_url=LLM_BASE_URL, timeout=60)

    counts = tag_counts()
    wanted = [t for t, c in counts.most_common() if c >= MIN_TRACKS]
    if args.limit:
        wanted = wanted[: args.limit]

    labels: dict[str, int] = {}
    if LABELS_PATH.exists():
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8")).get("labels", {})
    todo = [t for t in wanted if t not in labels]
    print(f"{len(wanted)} tags with >= {MIN_TRACKS} tracks; {len(todo)} still to judge")

    batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    start = time.time()

    def work(batch: list[str]):
        return batch, judge(client, batch)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, (batch, kept) in enumerate(pool.map(work, batches), start=1):
            if kept is None:
                print(f"  batch {done}/{len(batches)} failed, will be retried on the next run")
                continue
            labels.update({tag: int(tag in kept) for tag in batch})
            LABELS_PATH.write_text(
                json.dumps({"model": LLM_MODEL, "min_tracks": MIN_TRACKS, "labels": labels},
                           ensure_ascii=False, indent=0),
                encoding="utf-8",
            )
            print(f"  batch {done}/{len(batches)} done ({time.time() - start:.0f} s)", flush=True)

    kept = sum(labels.values())
    print(f"\n{len(labels)} tags judged, {kept} kept as descriptive ({100 * kept / max(len(labels), 1):.0f}%)")
    print("saved to", LABELS_PATH)


if __name__ == "__main__":
    main()

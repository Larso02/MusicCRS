"""Measure how well the LLM turns a feedback sentence into structured positives and negatives.

30 labelled sentences about a shown list of recommendations (real LLM, needs MY_OLLAMA_API_KEY).

Usage, from the repo root:
    python -m scripts.eval_feedback            # every sentence
    python -m scripts.eval_feedback --quiet    # summary and wrong answers only

A sentence counts as correct when every expected item is found on the right side (liked or rejected)
and none of the expected items shows up on the opposite side. "any_of" lists acceptable readings
for sentences that can reasonably be read in more than one way; for example "the third and fourth are
perfect" may be read as a select (the songs are added, which registers them as liked), and a
recommend with a hint ("more upbeat") counts as a stated wish.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "musiccrs"))

from router import Router  # noqa: E402

LLM_BASE_URL = "https://openwebui.ux.uis.no/api"
LLM_MODEL = "gorina10.gemma-4-31b"

RECS = ["Sparks by Coldplay", "Trouble by Coldplay", "Iris by The Goo Goo Dolls",
        "In the End by Linkin Park", "Scar Tissue by Red Hot Chili Peppers"]
STATE = {"playlist": ["Yellow by Coldplay"], "last": "Yellow by Coldplay",
         "open_list": RECS, "last_recommendations": RECS}

# (sentence, expectation). Keys: liked_positions / disliked_positions (sets of numbers);
# liked_tags, disliked_tags, liked_artists, disliked_artists (substrings that must be in that list);
# wants (substring that must be in wish or liked_tags); variety; any_of (list of alternative expectations)
CASES = [
    ("I like the first two, but the others are too heavy", {"liked_positions": {1, 2}}),
    ("I love Coldplay, more of that", {"liked_artists": ["coldplay"]}),
    ("no more Linkin Park please", {"disliked_artists": ["linkin"]}),
    ("too much metal", {"disliked_tags": ["metal"]}),
    ("less metal, more alternative rock", {"disliked_tags": ["metal"], "wants": "alternative rock"}),
    ("I want something more upbeat", {"wants": "upbeat"}),
    ("number 3 is great", {"liked_positions": {3}}),
    ("I hate the last one", {"disliked_positions": {5}}),
    ("not a fan of the second one", {"disliked_positions": {2}}),
    ("the third and fourth are perfect", {"any_of": [{"liked_positions": {3, 4}}, {"selects": {3, 4}}]}),
    ("make it more varied", {"variety": "more"}),
    ("more of the same please", {"variety": "less"}),
    ("give me more variety, I'm bored of Coldplay", {"variety": "more", "disliked_artists": ["coldplay"]}),
    ("I like Linkin Park but not Coldplay", {"liked_artists": ["linkin"], "disliked_artists": ["coldplay"]}),
    ("too slow, I want faster songs", {"disliked_tags": ["slow"], "wants": "fast"}),
    ("more energetic stuff", {"wants": "energetic"}),
    ("this is too sad", {"disliked_tags": ["sad"]}),
    ("I'm into grunge lately", {"wants": "grunge"}),
    ("no rap please", {"disliked_tags": ["rap"]}),
    ("love the 90s vibe", {"wants": "90s"}),
    ("I prefer female vocalists", {"wants": "female"}),
    ("the first one is great but the others are boring", {"liked_positions": {1}, "disliked_positions": {2, 3, 4, 5}}),
    ("skip the Red Hot Chili Peppers one", {"any_of": [{"disliked_positions": {5}}, {"disliked_artists": ["red hot"]}, {"selects": {1, 2, 3, 4}}]}),
    ("yes, more like Iris", {"any_of": [{"liked_positions": {3}}, {"liked_artists": ["goo goo"]}]}),
    ("can you play something calmer", {"wants": "calm"}),
    ("less pop", {"disliked_tags": ["pop"]}),
    ("I like all of them", {"any_of": [{"liked_positions": {1, 2, 3, 4, 5}}, {"selects": {1, 2, 3, 4, 5}}]}),
    ("I don't like any of these", {"disliked_positions": {1, 2, 3, 4, 5}}),
    ("more rock, less electronic", {"wants": "rock", "disliked_tags": ["electronic"]}),
    ("ugh, too many Coldplay songs", {"any_of": [{"disliked_artists": ["coldplay"]}, {"variety": "more"}]}),
]


def lower_list(value) -> list[str]:
    return [str(v).lower() for v in value] if isinstance(value, list) else []


def positions(value) -> set[int]:
    try:
        return {int(v) for v in value} if isinstance(value, list) else set()
    except (TypeError, ValueError):
        return set()


def selected(out: dict, n: int = 5) -> set[int] | None:
    """The positions a 'select' action takes from the list (None if it is not a select)."""
    if out.get("action") != "select":
        return None
    chosen = positions(out.get("positions"))
    if str(out.get("mode") or "").lower() == "all" or (not chosen and out.get("except")):
        chosen = set(range(1, n + 1))
    return chosen - positions(out.get("except"))


def holds(out: dict, expect: dict) -> bool:
    if "any_of" in expect:
        return any(holds(out, alt) for alt in expect["any_of"])
    if "selects" in expect:                      # adding the songs also counts as liking them
        return selected(out) == expect["selects"]
    liked_pos, disliked_pos = positions(out.get("liked_positions")), positions(out.get("disliked_positions"))
    for key, side, opposite in (("liked_positions", liked_pos, disliked_pos), ("disliked_positions", disliked_pos, liked_pos)):
        if key in expect and not (expect[key] <= side and not (expect[key] & opposite)):
            return False
    for key, opposite_key in (("liked_tags", "disliked_tags"), ("disliked_tags", "liked_tags"),
                              ("liked_artists", "disliked_artists"), ("disliked_artists", "liked_artists")):
        for needle in expect.get(key, []):
            if not any(needle in v for v in lower_list(out.get(key))):
                return False
            if any(needle in v for v in lower_list(out.get(opposite_key))):
                return False
    if "wants" in expect:
        pool = lower_list(out.get("liked_tags")) + [str(out.get("wish") or "").lower(), str(out.get("hint") or "").lower()]
        if not any(expect["wants"] in v for v in pool):
            return False
    if "variety" in expect and str(out.get("variety") or "").lower() != expect["variety"]:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    router = Router(OpenAI(api_key=os.getenv("MY_OLLAMA_API_KEY"), base_url=LLM_BASE_URL), LLM_MODEL)

    correct, times, wrong = 0, [], []
    for sentence, expect in CASES:
        start = time.time()
        out = router.route(sentence, STATE)
        times.append(time.time() - start)
        good = out is not None and out.get("action") in ("feedback", "recommend", "select") and holds(out, expect)
        correct += good
        if not good:
            wrong.append((sentence, out))
        if not args.quiet:
            print(f"{'OK ' if good else 'BAD'} {times[-1]:4.1f}s  {sentence!r:62} -> {json.dumps(out, ensure_ascii=False)[:150]}")

    n = len(CASES)
    print(f"\ncorrect {correct}/{n} = {100 * correct / n:.0f}% (the proposal promises at least 85%); median {sorted(times)[n // 2]:.1f} s")
    for sentence, out in wrong:
        print(f"  wrong: {sentence!r} -> {out}")


if __name__ == "__main__":
    main()

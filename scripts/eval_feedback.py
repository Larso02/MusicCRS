"""Measure how well the LLM turns a feedback sentence into structured positives and negatives.

Two sets of 30 labelled sentences about a shown list of recommendations (--set tuned / --set heldout) (real LLM, needs MY_OLLAMA_API_KEY).

Usage, from the repo root:
    python -m scripts.eval_feedback            # every sentence
    python -m scripts.eval_feedback --quiet    # summary and wrong answers only

A sentence counts as correct when every expected item is found on the right side (liked or rejected)
and none of the expected items shows up on the opposite side. "any_of" lists acceptable readings
for sentences that can reasonably be read in more than one way. Two equivalence rules, added after the first
look at the results and applied to every sentence: a select of songs counts as "liked these" (in the
chat, adding a suggested song registers a like), and a recommend with a hint ("more upbeat") counts as
a stated wish.
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


# A second set written AFTER the prompt was tuned on CASES, with the acceptable readings fixed before the
# first run. Nothing in the prompt has been changed because of these sentences. (Written by the author of
# the prompt, so it is not independent: add sentences written by someone else to make it a real test.)
HELDOUT = [
    ("honestly the first one is a banger", {"liked_positions": {1}}),
    ("ugh, Linkin Park again?", {"any_of": [{"disliked_artists": ["linkin"]}, {"disliked_positions": {4}}]}),
    ("can we get away from the sad stuff", {"disliked_tags": ["sad"]}),
    ("love the Red Hot Chilli Peppers", {"any_of": [{"liked_artists": ["red hot"]}, {"liked_positions": {5}}]}),
    ("number two and number five are not my thing", {"disliked_positions": {2, 5}}),
    ("i like 3 and 4", {"liked_positions": {3, 4}}),
    ("more acoustic please", {"wants": "acoustic"}),
    ("too mainstream, give me something less known", {"any_of": [
        {"wants": "less known"}, {"wants": "underground"}, {"wants": "obscure"}, {"wants": "indie"}, {"wants": "unknown"},
        {"disliked_tags": ["mainstream"]}, {"disliked_tags": ["pop"]}]}),
    ("give me a wider range of artists", {"variety": "more"}),
    ("all of these sound the same", {"variety": "more"}),
    ("Iris is perfect, exactly what I wanted", {"any_of": [{"liked_positions": {3}}, {"liked_artists": ["goo goo"]}]}),
    ("none of the coldplay songs please", {"disliked_artists": ["coldplay"]}),
    ("I'm in the mood for jazz", {"wants": "jazz"}),
    ("dont like the second, fourth and fifth", {"disliked_positions": {2, 4, 5}}),
    ("only the last two are good", {"liked_positions": {4, 5}}),
    ("something heavier would be nice", {"wants": "heavy"}),
    ("less emo", {"disliked_tags": ["emo"]}),
    ("i prefer the 80s sound", {"wants": "80s"}),
    ("the Goo Goo Dolls are great, but skip the rest", {"any_of": [
        {"liked_positions": {3}}, {"liked_artists": ["goo goo"]}]}),
    ("the vocals are too screamy", {"any_of": [{"disliked_tags": [x]} for x in ("scream", "metal", "hard", "heavy", "punk")]}),
    ("more songs from the 2000s", {"any_of": [{"wants": "2000"}, {"wants": "00s"}]}),
    ("I want to hear more female singers", {"wants": "female"}),
    ("love them all", {"any_of": [{"liked_positions": {1, 2, 3, 4, 5}}, {"selects": {1, 2, 3, 4, 5}}]}),
    ("Scar Tissue was a mistake", {"disliked_positions": {5}}),
    ("drop the Linkin Park and give me something calmer", {"any_of": [
        {"disliked_artists": ["linkin"], "wants": "calm"}, {"disliked_positions": {4}, "wants": "calm"}]}),
    ("too much guitar, more electronic", {"disliked_tags": ["guitar"], "wants": "electronic"}),
    ("mix it up", {"variety": "more"}),
    ("stick to this style", {"variety": "less"}),
    ("I really enjoy Coldplay but not Trouble", {"liked_artists": ["coldplay"], "disliked_positions": {2}}),
    ("hmm, the first one's fine I guess, the rest not so much", {"liked_positions": {1}, "disliked_positions": {2, 3, 4, 5}}),
]
# A third set, written after the prompt and the checker were final, with the acceptable readings fixed before the
# single run. The prompt must not be changed because of these sentences. (Written by the author of the prompt;
# sentences written by someone else would be a stronger test.)
FRESH = [
    ("the second song is amazing", {"liked_positions": {2}}),
    ("not feeling the first two", {"disliked_positions": {1, 2}}),
    ("I'd rather not hear any more Linkin Park", {"disliked_artists": ["linkin"]}),
    ("bring on the punk", {"wants": "punk"}),
    ("kinda bored of this, something fresh", {"variety": "more"}),
    ("just more of what you gave me", {"variety": "less"}),
    ("good picks, especially the third", {"liked_positions": {3}}),
    ("can you avoid anything too aggressive", {"any_of": [{"disliked_tags": [x]} for x in ("aggress", "metal", "hard", "heavy")]}),
    ("more piano", {"wants": "piano"}),
    ("I like the vibe of Iris, more like that", {"any_of": [{"liked_positions": {3}}, {"liked_artists": ["goo goo"]}, {"wants": "iris"}]}),
    ("fewer Coldplay songs", {"any_of": [{"disliked_artists": ["coldplay"]}, {"variety": "more"}]}),
    ("nah, the fourth one ruins it", {"disliked_positions": {4}}),
    ("give me upbeat summer songs", {"any_of": [{"wants": "upbeat"}, {"wants": "summer"}]}),
    ("something to relax to", {"any_of": [{"wants": "relax"}, {"wants": "calm"}, {"wants": "chill"}]}),
    ("love Red Hot Chili Peppers and Coldplay", {"liked_artists": ["coldplay", "red hot"]}),
    ("dislike the last one and the first one", {"disliked_positions": {1, 5}}),
    ("perfect! the first, third and fifth", {"liked_positions": {1, 3, 5}}),
    ("i prefer instrumental music", {"wants": "instrumental"}),
    ("less pop and more rock", {"disliked_tags": ["pop"], "wants": "rock"}),
    ("all five are great", {"liked_positions": {1, 2, 3, 4, 5}}),
    ("none of these, try again", {"any_of": [{"disliked_positions": {1, 2, 3, 4, 5}}, {"selects": set()}]}),
    ("the Goo Goo Dolls one is the best", {"any_of": [{"liked_positions": {3}}, {"liked_artists": ["goo goo"]}]}),
    ("make them sound more modern", {"wants": "modern"}),
    ("too repetitive", {"variety": "more"}),
    ("stay with the rock sound", {"any_of": [{"variety": "less"}, {"wants": "rock"}]}),
    ("I can't stand Trouble", {"disliked_positions": {2}}),
    ("more like Sparks", {"any_of": [{"liked_positions": {1}}, {"wants": "sparks"}]}),
    ("keep the energy high", {"any_of": [{"wants": "energ"}, {"wants": "upbeat"}, {"wants": "high"}]}),
    ("a bit slower please", {"any_of": [{"wants": "slow"}, {"wants": "calm"}, {"disliked_tags": ["fast"]}]}),
    ("I enjoy the Chili Peppers one but the Linkin Park one is too heavy", {"any_of": [
        {"liked_positions": {5}, "disliked_positions": {4}}, {"liked_artists": ["red hot"], "disliked_artists": ["linkin"]},
        {"liked_positions": {5}, "disliked_tags": ["heavy"]}, {"liked_artists": ["red hot"], "disliked_tags": ["heavy"]}]}),
]
SETS = {"tuned": CASES, "heldout": HELDOUT, "fresh": FRESH}


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


def mentions(text: str, needle: str) -> bool:
    """needle occurs in text, comparing words by their first four letters ("heavier" matches "heavy")."""
    if needle in text:
        return True
    stems = [w[:4] for w in needle.split()]
    words = [w[:4] for w in text.replace(",", " ").split()]
    return all(s in words for s in stems)


def holds(out: dict, expect: dict) -> bool:
    if "any_of" in expect:
        return any(holds(out, alt) for alt in expect["any_of"])
    if "selects" in expect:                      # adding the songs also counts as liking them
        return selected(out) == expect["selects"]
    if out.get("action") == "select":            # equivalence rule: adding a suggested song counts as liking it,
        out = {**out, "liked_positions": sorted(selected(out) or [])}      # so a select equals "liked these"
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
        if not any(mentions(v, expect["wants"]) for v in pool):
            return False
    if "variety" in expect and str(out.get("variety") or "").lower() != expect["variety"]:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--set", choices=sorted(SETS), default="tuned",
                        help="tuned: the sentences the prompt was tuned on; heldout: a second development set; fresh: the final test, run once")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    router = Router(OpenAI(api_key=os.getenv("MY_OLLAMA_API_KEY"), base_url=LLM_BASE_URL), LLM_MODEL)

    correct, times, wrong = 0, [], []
    cases = SETS[args.set]
    for sentence, expect in cases:
        start = time.time()
        out = router.route(sentence, STATE)
        times.append(time.time() - start)
        good = out is not None and out.get("action") in ("feedback", "recommend", "select") and holds(out, expect)
        correct += good
        if not good:
            wrong.append((sentence, out))
        if not args.quiet:
            print(f"{'OK ' if good else 'BAD'} {times[-1]:4.1f}s  {sentence!r:62} -> {json.dumps(out, ensure_ascii=False)[:150]}")

    n = len(cases)
    print(f"\ncorrect {correct}/{n} = {100 * correct / n:.0f}% (the proposal promises at least 85%); median {sorted(times)[n // 2]:.1f} s")
    for sentence, out in wrong:
        print(f"  wrong: {sentence!r} -> {out}")


if __name__ == "__main__":
    main()

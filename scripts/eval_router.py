"""Measure how well the LLM router understands messages (real LLM, needs MY_OLLAMA_API_KEY).

Usage, from the repo root:
    python -m scripts.eval_router            # prints every message and the summary
    python -m scripts.eval_router --quiet    # summary and wrong answers only

Each case is a message, the conversation state, the expected action and a few expected fields
(substrings, case-insensitive). The cases are the regression test for changes to the prompt.
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

EMPTY: dict = {}
PL = {"playlist": ["Yellow - Coldplay", "Clocks - Coldplay", "Smells Like Teen Spirit - Nirvana"],
      "last": "Yellow - Coldplay"}
LIST5 = dict(PL, open_list=["Sparks - Coldplay", "Trouble - Coldplay", "Iris - The Goo Goo Dolls",
                            "In the End - Linkin Park", "Scar Tissue - Red Hot Chili Peppers"])
CONFIRM = dict(PL, pending_confirm="Replace the current playlist with a new one for 'the gym'?")

# (message, state, expected action, {field: expected substring})
CASES = [
    ("add Yellow by Coldplay", EMPTY, "add", {"title": "yellow", "artist": "coldplay"}),
    ("could you throw stnad by me on there", EMPTY, "add", {"title": "stand by me"}),
    ("I want to hear some Radiohead, creep please", EMPTY, "add", {"title": "creep", "artist": "radiohead"}),
    ("add Love", EMPTY, "add", {"title": "love"}),
    ("get rid of the coldplay songs", PL, "remove", {"artist": "coldplay"}),
    ("drop clocks", PL, "remove", {"title": "clocks"}),
    ("remove the song by nirvana", PL, "remove", {"artist": "nirvana"}),
    ("take the second one off", PL, "remove", {"title": "clocks"}),
    ("what do I have so far?", PL, "view", {}),
    ("show me my songs", PL, "view", {}),
    ("let's start from scratch", PL, "clear", {}),
    ("nuke it all", PL, "clear", {}),
    ("what can you do?", EMPTY, "help", {}),
    ("how do I take a song off?", EMPTY, "help", {}),
    ("who sings this?", PL, "ask", {"kind": "artist", "subject": "yellow by coldplay"}),
    ("tell me who made it", PL, "ask", {"kind": "artist", "subject": "yellow by coldplay"}),
    ("who made the second one?", PL, "ask", {"kind": "artist", "subject": "clocks by coldplay"}),
    ("what genre is Yellow by Coldplay?", EMPTY, "ask", {"kind": "tags", "subject": "yellow by coldplay"}),
    ("how long is Clocks by Coldplay", PL, "ask", {"kind": "length", "subject": "clocks"}),
    ("what year did Nirvana's Smells Like Teen Spirit come out", PL, "ask", {"kind": "year"}),
    ("how many songs do you have by Muse", EMPTY, "ask", {"kind": "count", "subject": "muse"}),
    ("tell me about Radiohead", EMPTY, "ask", {"kind": "about", "subject": "radiohead"}),
    ("what's Coldplay's biggest hit", EMPTY, "ask", {"kind": "top", "subject": "coldplay"}),
    ("recommend something", PL, "recommend", {}),
    ("give me a few songs like these", PL, "recommend", {}),
    ("suggest something more mellow", PL, "recommend", {"hint": "mellow"}),
    ("1 and 3", LIST5, "select", {}),
    ("add the first two", LIST5, "select", {}),
    ("everything except the Linkin Park one", LIST5, "select", {}),
    ("the ones by coldplay", LIST5, "select", {}),
    ("none of them", LIST5, "select", {}),
    ("nah, skip those", LIST5, "select", {}),
    ("make me a playlist for a road trip", PL, "generate", {"description": "road trip"}),
    ("I'd like a calm evening playlist with some jazz", EMPTY, "generate", {"description": "jazz"}),
    ("sad love songs please", EMPTY, "generate", {}),
    ("yes", CONFIRM, "confirm", {"answer": "yes"}),
    ("no keep what I have", CONFIRM, "confirm", {"answer": "no"}),
    ("I like the first two but the rest are too heavy", LIST5, "feedback", {}),
    ("less metal, more alternative rock", PL, "feedback", {}),
    ("thanks, that's great!", PL, "chat", {}),
    ("hello", EMPTY, "chat", {}),
    ("what's the weather like?", EMPTY, "chat", {}),
    ("ok I'm done, bye", PL, "quit", {}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    router = Router(OpenAI(api_key=os.getenv("MY_OLLAMA_API_KEY"), base_url=LLM_BASE_URL), LLM_MODEL)

    correct, times, wrong = 0, [], []
    for message, state, action, fields in CASES:
        start = time.time()
        out = router.route(message, state)
        times.append(time.time() - start)
        flat = json.dumps(out, ensure_ascii=False).lower()
        good = out is not None and out["action"] == action and all(v in flat for v in fields.values())
        correct += good
        if not good:
            wrong.append((message, action, out))
        if not args.quiet:
            print(f"{'OK ' if good else 'BAD'} {times[-1]:4.1f}s  {message!r:58} -> {flat[:110]}")

    times.sort()
    n = len(CASES)
    print(f"\ncorrect {correct}/{n}; median {times[n // 2]:.1f} s, slowest {times[-1]:.1f} s")
    for message, action, out in wrong:
        print(f"  wrong: {message!r} expected {action}, got {out}")


if __name__ == "__main__":
    main()

import re

try:
    from track_database import norm
except ImportError:
    from musiccrs.track_database import norm

CAPABILITIES = {
    "add": {
        "available": True,
        "keywords": {"add", "put", "queue", "insert", "include", "append"},
        "title": "Add songs",
        "summary": "add a song by name, with or without the artist",
        "text": (
            "Say <i>add Yellow by Coldplay</i>, <i>add Coldplay: Yellow</i> or just <i> add Yellow</i>. Small typos are fine. If several songs share a title, "
            "I show them as buttons and you just pick one."
        ),
    },
        "remove": {
        "available": True,
        "keywords": {"remove", "delete", "drop", "undo"},
        "title": "Remove songs",
        "summary": "take a song off the playlist",
        "text": (
            "Say <i>remove Yellow</i> or <i>remove Yellow by Coldplay</i>, or use the "
            "Remove button next to the song in the playlist panel."
        ),
    },
    "view": {
        "available": True,
        "keywords": {"show", "view", "see", "list", "contains"},
        "title": "Show the playlist",
        "summary": "show what is on your playlist",
        "text": "Say <i>show my playlist</i> (it is also always visible in the panel).",
    },
    "clear": {
        "available": True,
        "keywords": {"clear", "empty", "reset"},
        "title": "Clear the playlist",
        "summary": "start over with an empty playlist",
        "text": "Say <i>clear my playlist</i>, or press the Clear button.",
    },
    "choose": {
        "available": True,
        "keywords": {"choose", "pick", "select", "ambiguous", "options", "several"},
        "title": "Choose between songs",
        "summary": "pick the right song when several match",
        "text": (
            "When I ask which song you mean, click a button, type its number, say "
            "<i>the second one</i>, or name the artist. Say <i>cancel</i> to skip."
        ),
    },
        "ask": {
        "available": True,
        "keywords": {"ask", "question", "questions", "info", "information"},
        "title": "Answer questions",
        "summary": "answer questions about songs and artists",
        "text": (
            "About a song: <i>who sings Yellow?</i>, <i>what album is Yellow on?</i>, "
            "<i>when was Yellow released?</i>, <i>how long is Yellow?</i>, "
            "<i>how popular is Yellow?</i>, <i>what genre is Yellow?</i>. "
            "About an artist: <i>what songs does Coldplay have?</i>, "
            "<i>what albums does Coldplay have?</i>, <i>how many songs does Coldplay have?</i>, "
            "<i>what is Coldplay's most popular song?</i>. "
            "After you add or ask about a song, <i>who sings it?</i> works too."
        ),
    },
        "recommend": {
        "available": True,
        "keywords": {"recommend", "suggest", "recommendation", "recommendations"},
        "title": "Recommend songs",
        "summary": "suggest 3-5 songs that fit your playlist",
        "text": "Say <i>recommend something</i> and I suggest songs related to your playlist.",
    },
}

_HELP_RE = re.compile(
    r"^/?help\b"
    r"|\bwhat (?:can|could) (?:you|i)\b"
    r"|\bwhat do you do\b"
    r"|\bhow (?:do|can|does|would|should|could) (?:i|you|we|this|one)\b"
    r"|\bhow to\b"
    r"|\bwhat (?:are|is) (?:your|the|all)\b.*\b(?:commands?|features?|options|capabilities)\b"
    r"|\bwhat commands\b"
    r"|\b(?:instructions|capabilities|features)\b",
    re.I,
)

def is_help_request(text: str) -> bool:
    return bool(_HELP_RE.search(text.strip()))

def _topics(text: str) -> list[str]:
    tokens = set(norm(text).split())
    return [
        name for name, cap in CAPABILITIES.items()
        if cap["available"] and cap["keywords"] & tokens
    ]

def help_answer(text: str) -> str:
    topics = _topics(text)
    if topics:
        return "<br>".join(f"<b>{CAPABILITIES[t]['title']}</b>: {CAPABILITIES[t]['text']}" for t in topics)
    
    lines = [
        f"\u2022 <b>{c['title']}</b>: {c['summary']}" for c in CAPABILITIES.values() if c["available"]
    ]
    return (
        "I'm MusicCRS. I help you build a playlist from the TalkPlay catalog. "
        "Here is what I can do:<br>"
        + "<br>".join(lines)
        + "<br>Ask for details, for example <i>how do I add a song?</i>" 
    )
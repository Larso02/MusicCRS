import re

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
# "one" is left out: "the second one" must not also select number 1
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}

FILLER = {
    "add", "put", "queue", "pick", "choose", "select", "take", "want", "like", "please", "pls",
    "the", "a", "an", "number", "numbers", "no", "nr", "option", "options", "song", "songs",
    "track", "tracks", "one", "ones", "them", "those", "these", "it", "of", "to", "my", "playlist",
    "i", "id", "ill", "im", "i'd", "i'll", "i'm", "let's", "can", "could", "would", "you", "lets",
    "let", "us", "go", "with", "for", "me", "also", "just", "only", "then", "from", "list", "in",
    "on", "by", "is", "are", "be", "that", "this", "and", "give", "get", "hear", "play", "listen",
    "have", "too", "as", "well",
}

_EXCEPT = re.compile(
    r"\b(?:except(?:\s+for)?|but\s+not|but|other\s+than|apart\s+from|aside\s+from|besides|"
    r"excluding|without|minus|not)\b"
)
_ALL = re.compile(r"\b(?:all|everything|every\s+one|everyone|each|the\s+lot|both)\b")
_NUM = r"(\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)(?:st|nd|rd|th)?"
_RANGE = re.compile(rf"\b{_NUM}\s*(?:-|to|through|thru|until)\s*{_NUM}\b")
_COUNT = r"(\d+|two|three|four|five|six|seven|eight|nine|ten)"
_FIRST_N = re.compile(rf"\b(?:first|top)\s+{_COUNT}\b")
_LAST_N = re.compile(rf"\blast\s+{_COUNT}\b")
_TOKEN = re.compile(r"\d+(?:st|nd|rd|th)?|[a-z']+")

def _number(token: str) -> int | None:
    token = re.sub(r"(?<=\d)(st|nd|rd|th)$", "", token)
    if token.isdigit():
        return int(token)
    return ORDINALS.get(token) or NUMBER_WORDS.get(token)

def _resolve(part: str, n: int, labels: list[str] | None) -> set[int] | None:
    found: set[int] = set()
    if _ALL.search(part):
        if re.search(r"\bboth\b", part) and n != 2:
            return None
        found |= set(range(1, n+1))
        part = _ALL.sub(" ", part)

    for m in list(_RANGE.finditer(part)):
        lo, hi = _number(m.group(1)), _number(m.group(2))
        if lo is None or hi is None or not 1 <= lo <= hi <= n:
            return None
        found.update(range(lo, hi + 1))
    part = _RANGE.sub(" ", part)

    for pattern, from_start in ((_FIRST_N, True), (_LAST_N, False)):
        for m in list(pattern.finditer(part)):
            count = _number(m.group(1))
            if count is None or not 1 <= count <= n:
                return None
            positions = range(1, count + 1) if from_start else range(n - count + 1, n + 1)
            found.update(positions)
        part = pattern.sub(" ", part)

    leftover = []
    for token in _TOKEN.findall(part):
        if token == "last":
            found.add(n)
        elif token in FILLER:
            continue
        elif (value := _number(token)) is not None:
            if not 1 <= value <= n:
                return None
            found.add(value)
        else:
            leftover.append(token)

    if leftover:
        if labels is None:
            return None
        words = set(leftover)
        hits = {i for i, label in enumerate(labels, start=1) if words <= set(label.split())}
        if not hits:
            return None
        found |= hits
    return found

def parse_selection(text: str, n: int, labels: list[str] | None = None) -> list[int] | None:
    t = text.lower().replace("\u2019", "'")
    t = re.sub(r"(?<=\d)\s*[\u2013\u2014-]\s*(?=\d)", "-", t)
    t = re.sub(r"[^\w\s'-]", " ", t)
    t = re.sub(r"(?<![\d])-|-(?![\d])", " ", t)

    split = _EXCEPT.search(t)
    include, exclude = (t[: split.start()], t[split.end():]) if split else (t, "")

    chosen = _resolve(include, n, labels)
    if chosen is None:
        return None
    if split:
        removed = _resolve(exclude, n, labels)
        if removed is None or not removed:
            return None
        if not chosen:
            chosen = set(range(1, n + 1))
        chosen -= removed
    return sorted(chosen) or None
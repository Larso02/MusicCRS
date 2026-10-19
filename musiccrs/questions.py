import re
from dataclasses import dataclass

TRACK_KINDS = {"track_artist", "track_album", "track_year", "track_length", "track_popularity"}
ARTIST_KINDS = {"artist_songs", "artist_albums", "artist_count", "artist_top"}
EITHER_KINDS = {"tags", "about"} 

IT_WORDS = {"it", "this", "that", "this song", "that song", "this track", "that track",
            "the song", "this one", "that one"}

Q = r"(?P<q>.+?)"

@dataclass
class Question:
    kind: str
    subject: str

_RAW = [
    # ---------- about a song ----------
    ("track_artist", r"^(?:who|which artist|what artist|which band|what band)\s+(?:sings|sang|performs|performed|plays|played|made|wrote|recorded)\s+" + Q),
    ("track_artist", r"^who(?:'s|\s+is|\s+was)\s+(?:the\s+)?(?:artist|singer|band|performer)\s+(?:of|for|on|behind)\s+" + Q),
    ("track_artist", r"^who\s+(?:is|was)\s+" + Q + r"\s+by"),
    ("track_artist", r"^whose\s+(?:song|track)\s+(?:is|was)\s+" + Q),
    ("track_artist", r"^(?:what|which)\s+(?:artist|singer|band)\s+(?:is|was)\s+" + Q + r"\s+(?:by|from)"),

    ("track_album", r"^(?:what|which)\s+album\s+(?:is|was)\s+" + Q + r"\s+(?:on|from|in|part of|included on)"),
    ("track_album", r"^(?:what|which)\s+album\s+(?:does|did)\s+" + Q + r"\s+(?:belong to|come from|appear on|appear in|come on|come out on|feature on)"),
    ("track_album", r"^(?:what(?:'s|\s+is)\s+)?(?:the\s+)?album\s+(?:of|for)\s+" + Q),
    ("track_album", r"^(?:what|which)\s+album\s+(?:has|contains|includes)\s+" + Q),

    ("track_year", r"^(?:when|what\s+year)\s+(?:was|did|is)\s+" + Q + r"\s+(?:released|release|come out|made|written|recorded|out|published|first released)"),
    ("track_year", r"^(?:when|what\s+year)\s+" + Q + r"\s+(?:was|were)\s+(?:released|made|written|recorded|published)"),
    ("track_year", r"^(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:release date|release year|year|date)\s+(?:of|for)\s+" + Q),
    ("track_year", r"^how\s+old\s+is\s+" + Q),

    ("track_length", r"^how\s+(?:long|many minutes)\s+(?:is|was|does)\s+" + Q + r"(?:\s+(?:last|take|run|go))?"),
    ("track_length", r"^how\s+long\s+" + Q + r"\s+(?:is|was)"),
    ("track_length", r"^(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:length|duration|runtime|running time)\s+(?:of|for)\s+" + Q),

    ("track_popularity", r"^how\s+(?:popular|famous|well known)\s+(?:is|was)\s+" + Q),
    ("track_popularity", r"^(?:what(?:'s|\s+is)\s+)?(?:the\s+)?popularity\s+(?:of|for)\s+" + Q),
    ("track_popularity", r"^is\s+" + Q + r"\s+(?:popular|a hit|well known|famous|a big song)"),

    # ---------- song or artist ----------
    ("tags", r"^(?:what|which)\s+(?:genre|genres|style|styles|tags|mood|vibe|(?:kind|type) of (?:music|songs?))\s+(?:is|are|does|do|was)\s+" + Q + r"(?:\s+(?:have|belong to|fall under|count as|play|make|sing|produce|do|known for))?"),
    ("tags", r"^(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:genre|genres|style|styles|tags)\s+(?:of|for)\s+" + Q),
    ("about", r"^(?:tell me (?:more )?about|what do you know about|info(?:rmation)? (?:on|about)|describe)\s+" + Q),
    ("about", r"^who\s+(?:is|was|are)\s+" + Q),

    # ---------- about an artist ----------
    ("artist_songs", r"^(?:what|which)\s+(?:songs|tracks|music|hits)\s+(?:does|do|did|has|have)\s+" + Q + r"(?:\s+(?:have|sing|make|made|release|released|play|record|recorded|write|wrote))?"),
    ("artist_songs", r"^(?:what|which)\s+(?:songs|tracks|music|hits)\s+(?:by|from)\s+" + Q + r"(?:\s+(?:do you have|are there|do you know|is there))?"),
    ("artist_songs", r"^(?:list|show|name|give me)\s+(?:me\s+)?(?:some\s+|all\s+|the\s+)?(?:songs|tracks|music|hits)\s+(?:by|from|of)\s+" + Q),
    ("artist_songs", r"^(?:songs|tracks|music)\s+(?:by|from)\s+" + Q),
    ("artist_songs", r"^(?:what|which)\s+(?:does|did)\s+" + Q + r"\s+(?:sing|play|make|write|release|record)"),

    ("artist_albums", r"^(?:what|which)\s+albums\s+(?:does|do|did|has|have)\s+" + Q + r"(?:\s+(?:have|release|released|make|made|put out|record|recorded))?"),
    ("artist_albums", r"^(?:what|which)\s+albums\s+(?:by|from)\s+" + Q + r"(?:\s+(?:do you have|are there))?"),
    ("artist_albums", r"^(?:list|show|name|give me)\s+(?:me\s+)?(?:some\s+|all\s+|the\s+)?albums\s+(?:by|from|of)\s+" + Q),
    ("artist_albums", r"^albums\s+(?:by|from)\s+" + Q),

    ("artist_count", r"^how many\s+(?:songs|tracks)\s+(?:does|do|did|has|have)\s+" + Q + r"(?:\s+(?:have|got|released|made|in your catalog))?"),
    ("artist_count", r"^how many\s+(?:songs|tracks)\s+(?:by|from)\s+" + Q + r"(?:\s+(?:do you have|are there|are in your catalog|exist))?"),

    ("artist_top", r"^(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:most popular|best known|best-known|biggest|top|most famous)\s+(?:song|track|hit)\s+(?:by|of|from)\s+" + Q),
    ("artist_top", r"^(?:what(?:'s|\s+is)\s+)?" + Q + r"'s\s+(?:most popular|best known|best-known|biggest|top|most famous)\s+(?:song|track|hit)"),
    ("artist_top", r"^(?:which|what)\s+(?:song|track)\s+(?:by|from)\s+" + Q + r"\s+is\s+(?:the\s+)?(?:most popular|best known|biggest|most famous)"),
]

_PATTERNS =  [(kind, re.compile(pattern + r"\s*$", flags=re.I)) for kind, pattern in _RAW]

_LEAD = re.compile(
        r"^(?:please\s+)?(?:(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell\s+me|say|find\s+out)"
    r"|tell\s+me|do\s+you\s+know|i(?:'d|\s+would)\s+like\s+to\s+know|i\s+want\s+to\s+know"
    r"|i\s+wonder|let\s+me\s+know)\s+",
    re.I,
)

def parse_question(text: str) -> Question | None:
    t = text.strip().replace("\u2019", "'").rstrip(" ?!.")
    if not t :
        return None
    for variant in (t, _LEAD.sub("", t, count=1)):
        for kind, pattern in _PATTERNS:
            m = pattern.match(variant)
            if m:
                subject = m.group("q").strip(" \"'")
                if subject:
                    return Question(kind, subject)
    return None

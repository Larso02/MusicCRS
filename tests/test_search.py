"""The catalog search behind the search boxes in the web UI."""

import time

import pytest

from musiccrs.track_database import TrackDatabase

search_api = pytest.importorskip("musiccrs.search_api")


@pytest.fixture(scope="module")
def db():
    return TrackDatabase()


def names(tracks):
    return [f"{t['track_name']} - {', '.join(t['artist_name'])}" for t in tracks]


# ---------- the database search ----------
def test_results_have_what_the_ui_needs(db):
    track = db.search_catalog("yellow", 1)[0]
    assert set(track) == {"track_id", "track_name", "artist_name", "album_name", "popularity"}
    assert isinstance(track["artist_name"], list) and isinstance(track["album_name"], list)


def test_a_title_finds_songs_with_that_title_most_popular_first(db):
    results = db.search_catalog("yellow", 15)
    assert names(results)[0] == "Yellow - Coldplay"                   # exact title, most popular
    exact = [t for t in results if t["track_name"] == "Yellow"]
    assert [t["popularity"] for t in exact] == sorted((t["popularity"] for t in exact), reverse=True)
    assert all("yellow" in t["track_name"].lower() or "yellow" in " ".join(t["artist_name"]).lower() for t in results)


def test_an_exact_title_comes_before_longer_titles(db):
    results = db.search_catalog("clocks", 15)
    first_partial = next((i for i, t in enumerate(results) if t["track_name"] != "Clocks"), len(results))
    assert all(t["track_name"] == "Clocks" for t in results[:first_partial]) and first_partial >= 1


def test_an_artist_finds_their_songs_by_popularity(db):
    results = db.search_catalog("coldplay", 20)
    assert len(results) == 20 and all("Coldplay" in t["artist_name"] for t in results)
    assert [t["popularity"] for t in results] == sorted((t["popularity"] for t in results), reverse=True)


def test_words_can_be_split_between_title_and_artist(db):
    results = db.search_catalog("yellow coldplay", 5)
    assert names(results)[0] == "Yellow - Coldplay"
    assert all("Coldplay" in t["artist_name"] for t in results)
    assert db.search_catalog("coldplay yellow", 1) == db.search_catalog("yellow coldplay", 1)   # word order is free


def test_partial_words_work_while_typing(db):
    assert "Yellow - Coldplay" in names(db.search_catalog("yell cold", 10))


@pytest.mark.parametrize("text", ["YELLOW", "Yellow!", "  yellow  ", "yéllow"])
def test_case_punctuation_and_accents_do_not_matter(db, text):
    assert names(db.search_catalog(text, 1)) == ["Yellow - Coldplay"]


@pytest.mark.parametrize("text", ["", "   ", "!!!", "---"])
def test_nothing_to_search_for(db, text):
    assert db.search_catalog(text) == []


def test_unknown_words_find_nothing(db):
    assert db.search_catalog("zzzqqxx") == []


def test_the_limit_is_respected(db):
    assert len(db.search_catalog("love", 7)) == 7


def test_quotes_and_sql_are_just_text(db):
    assert db.search_catalog("'; DROP TABLE tracks; --") == []
    assert db.search_catalog("o'brien\" OR 1=1")  is not None
    assert len(db.search_catalog("yellow")) > 0                      # the table is still there


def test_only_the_first_words_are_used(db):
    long_query = " ".join(["yellow"] * 20)
    assert db.search_catalog(long_query, 3)                           # no error, still finds it


def test_a_search_is_fast_enough_to_type_into(db):
    start = time.time()
    for text in ("a", "the", "love you", "yellow col", "bohemian rhapsody queen"):
        db.search_catalog(text, 15)
    assert (time.time() - start) / 5 < 0.3


# ---------- search() with the fuzzy fallback ----------
class FakeFuzzy:
    def __init__(self, hits):
        self.hits, self.queries = hits, []

    def search(self, text, k=5):
        self.queries.append(text)
        return self.hits


def test_the_limit_is_clamped(db):
    assert len(search_api.search(db, "love", 1000)) == search_api.MAX_LIMIT
    assert len(search_api.search(db, "love", 0)) == 1


def test_close_spellings_are_offered_when_nothing_matches_exactly(db):
    yellow = db.search_catalog("yellow", 1)[0]["track_id"]
    fuzzy = FakeFuzzy([(yellow, 0.9), ("not-in-the-catalog", 0.95), (db.search_catalog("clocks", 1)[0]["track_id"], 0.3)])
    results = search_api.search(db, "yelow", 10, fuzzy)
    assert fuzzy.queries == ["yelow"]
    assert names(results) == ["Yellow - Coldplay"]                    # unknown ids and weak guesses are dropped


def test_fuzzy_is_never_mixed_into_real_matches(db):
    fuzzy = FakeFuzzy([])
    assert len(search_api.search(db, "yellow coldplay", 10, fuzzy)) >= 2 and fuzzy.queries == []
    few = db.search_catalog("yellow coldplay", 10)
    padded = search_api.search(db, "yellow coldplay", 10, FakeFuzzy([(db.search_catalog("clocks", 1)[0]["track_id"], 0.99)]))
    assert names(padded) == names(few)                                # one or two exact hits are not padded with guesses


def test_a_typo_finds_the_song_with_the_real_fuzzy_index(db, agent_module):
    results = search_api.search(db, "stnad by me", 10, agent_module.get_fuzzy_index(db))
    assert any(t["track_name"].lower() == "stand by me" for t in results)


# ---------- the HTTP route ----------
@pytest.fixture()
def client(db):
    flask = pytest.importorskip("flask")
    app = flask.Flask("test")
    search_api.register_search(app, db)

    @app.get("/other")
    def other():
        return "x"

    return app.test_client()


def test_the_route_returns_json_the_front_end_can_read(client):
    response = client.get("/search?q=yellow&limit=3")
    assert response.status_code == 200
    data = response.get_json()
    assert data["query"] == "yellow" and len(data["results"]) == 3
    assert data["results"][0]["track_name"] == "Yellow"
    assert response.headers["Access-Control-Allow-Origin"] == "*"      # the UI runs on another port


def test_the_route_handles_missing_and_bad_parameters(client):
    assert client.get("/search").get_json()["results"] == []
    assert client.get("/search?q=").get_json()["results"] == []
    assert len(client.get("/search?q=love&limit=abc").get_json()["results"]) == search_api.DEFAULT_LIMIT
    assert len(client.get("/search?q=love&limit=-5").get_json()["results"]) == 1


def test_other_routes_are_not_opened_to_other_origins(client):
    assert "Access-Control-Allow-Origin" not in client.get("/other").headers

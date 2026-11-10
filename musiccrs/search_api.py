"""A small HTTP search over the whole catalog, for the search box in the web UI.

    GET /search?q=yellow col&limit=15   ->   {"query": "...", "results": [{track_id, track_name,
                                              artist_name, album_name, popularity}, ...]}

The chat itself runs over the socket; a search box needs an answer on every keystroke without adding
messages to the conversation, so it uses this plain route next to the socket on the same Flask app.
"""

DEFAULT_LIMIT = 15
MAX_LIMIT = 50
FUZZY_MIN_SIMILARITY = 0.7  # how close a misspelled title has to be


def search(db, query: str, limit: int = DEFAULT_LIMIT, fuzzy_index=None) -> list[dict]:
    """Tracks matching the words of the query in title or artist, best first.

    Only when nothing matches exactly (a typo such as "stnad by me") and a fuzzy index is given,
    close spellings are offered instead. Weak guesses are never added to real matches.
    """
    limit = max(1, min(int(limit), MAX_LIMIT))
    results = db.search_catalog(query, limit)
    if fuzzy_index is not None and query.strip() and not results:
        guesses = [(i, s) for i, s in fuzzy_index.search(query, k=5) if s >= FUZZY_MIN_SIMILARITY]
        by_id = {t["track_id"]: t for t in db.get_tracks([i for i, _ in guesses])}
        results += [by_id[i] for i, _ in guesses if i in by_id]
    return results[:limit]


def register_search(app, db, fuzzy_index=None) -> None:
    """Add GET /search to a Flask app (and allow the front end, served from another port, to call it)."""
    from flask import jsonify, request

    @app.get("/search")
    def _search():
        query = request.args.get("q", "")
        try:
            limit = int(request.args.get("limit", DEFAULT_LIMIT))
        except ValueError:
            limit = DEFAULT_LIMIT
        results = search(db, query, limit, fuzzy_index)
        return jsonify({"query": query, "results": results})

    @app.after_request
    def _allow_front_end(response):
        if request.path == "/search":
            response.headers["Access-Control-Allow-Origin"] = "*"
        return response

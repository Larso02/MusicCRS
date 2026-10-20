# MusicCRS Part 2: what we built, how, and why

Status checked on 2026-10-20. All numbers below were measured on the real system
(47,071-track catalog, real LLM, real embeddings), not estimated.

## 1. Overview

MusicCRS is a web chat system (React front end, DialogueKit/Flask back end) that manages a playlist
over a catalog of 47,071 TalkPlay tracks. One agent class (`musiccrs/musiccrs.py`) receives every
user message and decides what to do. The feature code is split into small modules that can be tested
without the web server:

| Module | Job | Requirement |
|---|---|---|
| `scripts/import_tracks.py` | builds the SQLite catalog `data/tracks.db` | R2 |
| `musiccrs/track_database.py` | queries on the catalog, text normalisation | R2, R4, R5 |
| `musiccrs/fuzzy.py` | typo-tolerant track search | R2 |
| `musiccrs/capabilities.py` | the list of features and the help text | R3 |
| `musiccrs/questions.py` | recognises questions about songs and artists | R5 |
| `musiccrs/recommender.py` | "recommend" from the playlist | R6 |
| `musiccrs/selection.py` | "add the first two", "all except 2" | R7 |
| `musiccrs/playlist_generator.py` | a whole playlist from a description | R8 |
| `frontend/.../ChatBox.tsx` | chat, playlist panel, buttons | R1 |
| `retrieval/` (Part 1) | BM25 retriever, reused by R8 | R8 |

**Turn flow.** Each message goes through `receive_utterance`, which checks, in this order: a pending
yes/no question (R8), a pending "which one?" list (R4/R7), view, clear, generate, add, remove, help,
questions, recommend, and the old LLM commands. The first match answers. Every reply also carries the
current playlist as a dialogue act, so the UI panel always matches the server.

**Deterministic first.** R1-R7 use no LLM at all. Only R8 calls the LLM, once per request. This keeps
answers fast and predictable, and keeps us far below the limit of 5-10 LLM calls per turn.

## 2. Requirement by requirement

### R1: web chat and UI controls (1p)
- Chat with a playlist panel under it: numbered tracks, a **Remove** button per track, a **Clear**
  button and an **Add** box. The buttons send the same commands the chat understands (`/remove <id>`,
  `/clear`, `/add <text>`), so there is one code path for chat and UI.
- **Why:** one path means no state can drift apart. The server is the single source of truth; every
  reply carries the playlist (`PLAYLIST_STATE` act) and the panel simply renders it.
- Buttons for choices (R4/R6) come from an `OPTIONS` act; clicking one sends the label as text, so
  the pick goes through the same parser as typing.

### R2: add, remove, view, clear in natural language, with a database (3p)
- **Database.** `import_tracks.py` loads the catalog into SQLite: a `tracks` table with
  pre-normalised title/artist columns plus release date, duration and tags, and a `track_artists`
  table (one row per artist and track). Only ids from this table can enter the playlist; the
  playlist stores track ids, never free text. LLM output (R8) is also checked against the database.
- **Why SQLite:** it needs no server, loads in seconds, and an indexed lookup on the normalised title
  is instant.
- **Commands.** Regexes (`_ADD_RE`, `_REMOVE_RE`, `_VIEW_RE`, `_CLEAR_RE`) accept many phrasings:
  `add Yellow by Coldplay`, `put clocks on my playlist`, `remove the song by nirvana`,
  `what songs do I have?`, `start over`, `delete all songs`.
- **Loose references.** `_lookup(query, then)` is the single "find one track" routine:
  1. a raw track id (what the UI buttons send),
  2. exact title match, optionally narrowed by "by <artist>"; one hit is added directly,
  3. several different songs with that title go to the R4 "which one?" list,
  4. otherwise a typo-tolerant search (`fuzzy.py`): a character-trigram BM25 index gets 40
     candidates and `difflib` re-ranks them. A score of at least 0.9 is accepted with a note
     ("I read 'stnad by me' as 'Stand By Me - Ben E. King'"); 0.6-0.9 becomes a "Did you mean?" list.
  5. nothing close: "I couldn't find ...".
- **Why trigrams + difflib:** trigram BM25 finds candidates quickly across 47k titles even with
  swapped letters; `difflib` then orders the few candidates by how similar the whole string is.
  We tested this on typo'd titles before choosing the two thresholds.
- **Removal** matches words of the request against the title and artist of the playlist songs, so
  `remove nirvana` and `remove the song by nirvana` work. If several playlist songs match it asks
  which one (same list mechanism as R4) instead of guessing.
- **Duplicates** are never added twice ("already in your playlist").

### R3: help in natural language (1p)
- `capabilities.py` is a registry: for each feature a title, summary, keywords and an example text.
  `is_help_request` recognises "what can you do?", "how do I remove a song?", "help".
  `help_answer` replies with the whole list, or only the feature whose keywords match the question.
- **Why a registry:** the overview, the focused answers and the "available" flag all come from the
  same data, so adding a feature means adding one entry, and help can never advertise something that
  is not implemented.

### R4: disambiguation with intelligent ranking (1p)
- When several different songs match ("Love", "Yellow", "Iris"), the agent shows a numbered list with
  buttons. Duplicates of the same song (same title and artist on different albums) are collapsed.
- **Ranking** (`_rank_for_playlist`): `popularity/100 + 1.0 if the artist is already on the playlist
  + 0.5 if the album is`. A popular song, or one that fits what you already chose, comes first.
- **State machine.** The list is stored as `_pending` together with a callback `then(track, note)`
  and the text for "cancel". The user can answer with a number ("2"), an ordinal ("the second
  one"), the artist ("the one by Coldplay"), the button, or "cancel". A new command (for example
  "show playlist") closes the list and is executed normally, so the user is never trapped.
- **Why a callback:** the same list now serves four jobs - add a song, answer a question about it
  (R5), remove it (R2), and add recommendations (R6). Only `then` differs.

### R5: questions about tracks and artists (1p, needs 4 types)
- `questions.py` holds a table of about 40 regex patterns; `parse_question` returns a kind and a
  subject. **Songs (7 kinds):** artist, album, release date, length, popularity, tags/genre, summary.
  **Artists (6 kinds):** songs, albums, how many songs, most popular song, tags/style, summary.
  Tags and summary work for both, so there are 11 distinct kinds, well over the 4 required.
- Examples: `who sings Yellow?`, `when was Yellow released?`, `how long is Clocks?`,
  `what genre is Coldplay?`, `how many songs does Coldplay have?`, `tell me about Nirvana`.
- **Context:** `who sings it?` refers to the last song added or asked about (`_last_track`).
- **Artist matching** uses an artist table with a leading "the" removed ("The Beatles" = "Beatles")
  and `difflib` for typos; the answer says what it understood.
- **Honest answers:** popularity is shown as a score and as a percentile over the catalog; tags are
  cleaned of junk tags ("favorites", "seen live") and the artist's tags are counted over all their
  songs, which reads better than one song's noisy tags.
- **Why regexes and not the LLM:** instant (under 0.5 s), no cost, no hallucinations - every answer
  comes from the database. The price is that unusual wording is not understood; a new phrasing is
  one more line in the table.

### R6: recommend 3-5 songs related to the playlist (1p)
- `recommender.py` scores the whole catalog against the playlist and returns 5 songs, each with a
  reason ("same artist", "similar style: indie rock, alternative"). Score:

  `1.0 same artist + 0.5 same album + 1.0 same artist as the last song + 1.0 popularity
  + 1.0 tag similarity (TF-IDF cosine to the playlist's tags) + 0.5 release year close to the playlist`

- **Why these features:** the first four are the features that worked in our Part 1 retriever. Tags
  and release year were added because they gave a measurable gain. On 1,337 training turns ("predict
  the next song from the earlier ones"), nDCG@20 was 0.2205 for artist/album/popularity and 0.2317
  with tags and year, with weights chosen on one half and checked on the other.
- **What we tested and rejected:** BM25 over the playlist's metadata (no gain), and embedding
  centroids (clearly worse than the metadata features in our experiment).
- **Variety:** at most 2 songs per artist and at most 2 by artists already on the playlist, so the
  list is not just "more of the same". "recommend more" never repeats earlier suggestions.
- **Flow:** the five songs are shown as a pick-list (buttons plus text), so a recommendation becomes
  a playlist entry in one step.
- **Speed:** about 10-50 ms. The catalog features are built once at server start (about 3 s).

### R7: natural-language selection (1p, needs 3 formulations)
- `selection.py` has a pure function `parse_selection(text, n, labels)` that returns the chosen
  positions, or `None` if the text is not a selection. It supports:
  - numbers and lists: `1 and 3`, `1, 2, 5`, `number 2 and number 4`
  - ordinals and position words: `the first and the last`, `2nd and 4th`
  - quantities: `the first two`, `top three`, `the last 3`
  - ranges: `2-4`, `1 to 3`, `first through third`
  - all: `all`, `add everything`, `add them all`
  - exclusions: `all except 2`, `all but the second`, `add them except the one by Metallica`
  - by name: `the Coldplay ones`, `the Nirvana songs`
  - none: `none`, `nothing, thanks`, `no songs`, `cancel`
- **Safety rules:** a number outside the list is never guessed; a bare artist name that matches two
  songs asks again unless the user says "ones/songs/all"; in a question list ("when was Iris
  released?") picking several answers "that is about one song" instead of adding anything.
- **Why a parser module:** it is plain text in, positions out, so it is tested with about 60 phrasings
  and 20 non-selections without the agent. `_handle_choice` only maps positions to songs.
- Several songs are added by `_add_many`, which skips songs already on the playlist and says so.

### R8: a whole playlist from a description (1p)
- `make me a playlist for the gym`, `create a playlist with sad love songs`, `/generate ...`.
- **Steps.** (1) One LLM call turns the request into JSON: search keywords and a duration in minutes
  (the LLM picks a sensible length itself: gym 45-75 min, studying 60-120 min, road trip 90-180 min;
  we clamp to 15-120 min). (2) Candidates come from two retrievers fused with reciprocal rank
  fusion: our Part 1 BM25 retriever over names and tags, and a dense retriever (Qwen3 embedding of
  the keywords against the precomputed track-metadata embeddings). (3) Songs are added until the
  target duration is reached, skipping tracks with missing or implausible durations, at most 3 per
  artist and at most 40 songs. (4) Ids are verified against the database.
- **Length without asking:** the user is never asked for a number of songs; the duration target
  drives it, and the reply states the result ("13 songs, approximately 61 minutes").
- **Why fusion:** on 1,500 training requests, BM25 alone reached nDCG@20 0.0655 and BM25 + metadata
  embeddings 0.0829. BM25 finds the literal words, the embeddings find the meaning.
- **Why the prompt forbids occasion words:** BM25 matched "gym" in the band Gym Class Heroes. The
  prompt now tells the LLM to describe the music (genre, mood, tempo) and not the place or activity;
  themes the user names ("sad", "love", "rain") and artist names are kept. We also tested searching
  tags only, but that scored clearly worse on real requests (0.054 vs 0.073), so the retrieval
  itself stays unchanged.
- **Safe overwrite:** a generated playlist replaces the current one, so with a non-empty playlist the
  agent first asks "replaces your current playlist (N songs). Say yes or no" and does nothing until
  the answer. Any other message cancels the question.
- **Speed:** the BM25 index, the catalog and the embedding model are loaded at server start
  (about 30 s once). A request then takes about 2 s, mostly the LLM. If the embeddings cannot load,
  generation falls back to BM25 only.

## 3. Cross-cutting design choices

| Choice | Why |
|---|---|
| Rules and parsers first, LLM only for R8 | Fast (all replies under 3 s), no hallucinated songs, easy to test, far below the LLM-call limit |
| Playlist stores track ids only | Nothing outside the catalog can ever be added; the UI and the agent share one truth |
| One "pick from a list" mechanism with a `then` callback | R4, R5, R6, R2-remove all reuse it, so the UI buttons and the typed answers behave identically |
| Heavy objects built once per process, warmed in `__main__` | DialogueKit creates a new agent per user; indexes must not be rebuilt per connection |
| User messages that are real commands close an open list | The user is never stuck in a question |
| Replies escape user text (`html.escape`) | The UI renders replies as HTML |
| Everything verified by tests and an end-to-end script | 399 tests (parsers, flows, the real catalog) plus a scripted conversation over R1-R8 with timings |

## 4. Evidence

- **Part 1 retriever:** final score 0.2857 on the test split against the target 0.22; the same
  Part 1 feature ideas (artist, album, last artist, popularity) drive R6, and the retriever drives R8.
- **End-to-end run (real LLM, real embeddings):** every turn under 3 s; slowest were R8 requests at
  about 2 s; every other turn (adds, removes, questions, recommendations, selections) took
  under 0.5 s. The server warm-up takes about 30 s once.
- **Tests:** `pytest` runs 399 tests covering every requirement (the catalog tests use the real
  database).

## 5. Known limits (say them before the examiner finds them)

- Question and command recognition is pattern based; very unusual wording gets "I don't understand".
- Search keywords can still match song titles literally ("High Fidelity" for "high energy").
- Tags in the data are noisy; artist-level tags are better than song-level tags.
- Mood and activity playlists were judged by reading the results, not with a quality metric.
- `one` is not read as the number 1 in selections ("one and three"); use `1 and 3` or `first and third`.
- The first server start downloads about 1.2 GB (embedding model) and 0.8 GB (track embeddings).

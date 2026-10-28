# MusicCRS: what we built, how, and why

Status checked on 2026-10-28. All numbers below were measured on the real system
(47,071-track catalog, real LLM, real embeddings), not estimated.

## 1. Overview

MusicCRS is a web chat system (React front end, DialogueKit/Flask back end) that manages a playlist
over a catalog of 47,071 TalkPlay tracks. One agent class (`musiccrs/musiccrs.py`) receives every
user message and decides what to do. The feature code is split into small modules that can be tested
without the web server:

| Module | Job | Requirement |
|---|---|---|
| `scripts/import_tracks.py` | builds the SQLite catalog `data/tracks.db` | R2 |
| `musiccrs/track_database.py` | queries on the catalog, text normalisation, tag helpers | R2, R4, R5 |
| `musiccrs/router.py` | the LLM reads the message and returns one action | all |
| `musiccrs/fuzzy.py` | typo-tolerant track search | R2 |
| `musiccrs/capabilities.py` | the list of features and the help text | R3 |
| `musiccrs/questions.py` | rule-based recognition of questions (fallback) | R5 |
| `musiccrs/recommender.py` | "recommend" from the playlist | R6 |
| `musiccrs/selection.py` | rule-based "the first two", "all except 2" (fallback) | R7 |
| `musiccrs/playlist_generator.py` | a whole playlist from a description | R8 |
| `scripts/label_tags.py` + `data/tag_labels.json` | the LLM judges which tags describe music | R5, R6 |
| `frontend/.../ChatBox.tsx` | chat, playlist panel, buttons | R1 |
| `retrieval/` (Part 1) | BM25 retriever, reused by R8 | R8 |

## 2. How a message is handled: LLM first, rules as the fallback

Every message goes through `receive_utterance`:

1. **Exact inputs skip the LLM.** `/commands` (the UI buttons use them) and a click on an option
   button are exact, so they go straight to the code.
2. **The LLM understands the message.** `router.py` sends the message plus a short state summary
   (numbered playlist, last song, open pick-list, pending yes/no question) and gets back one JSON
   action: `add`, `remove`, `view`, `clear`, `help`, `ask`, `recommend`, `select`, `generate`,
   `confirm`, `feedback`, `chat` or `quit`, with fields such as title, artist, positions.
3. **Our code carries the action out.** Each action calls the existing, tested functions
   (`_lookup`, `_add_many`, `_answer_question`, `_recommend`, ...). The LLM never supplies songs or
   facts, so only database songs can reach the playlist.
4. **Rules are the safety net.** If there is no LLM, the call takes more than 4 s, fails, or returns
   something unusable (bad JSON, unknown action, missing fields, a number outside the open list),
   the rule-based parsers (regular expressions) answer instead. Nothing needs the LLM to work.

**Why this design**
- A rule table can only recognise wordings someone wrote down. The LLM handles new phrasings
  ("could you throw stnad by me on there", "nuke it all", "who made the second one?") and resolves
  references from context ("the second one", "yes") because it sees the state.
- It stays safe: the model chooses *what to do*, the database decides *which songs exist*.
- It stays fast: the model answers in 0.5 s (median) on the 43-message test; one call per message,
  far below the limit of 5-10 LLM calls per turn.
- It stays testable: unit tests replace the LLM with a script, and `scripts/eval_router.py`
  measures the real model.

**Measured.** 43 labelled messages (add, remove, view, clear, help, five kinds of questions,
recommend, selection from an open list, generate, yes/no, feedback, chat, quit): **43 of 43** correct
action and key fields, median 0.5 s, slowest 1.6 s. An earlier version dropped the artist from song
names ("Yellow" instead of "Yellow by Coldplay"); the prompt now asks for "Title by Artist", which
fixed it. In the live conversation every turn stayed under 3 s.

**Open points (Part 4).** `recommend` understands a hint ("more mellow") but does not use it yet,
and `feedback` only records the wish and recommends. Turning feedback into a preference profile with
Rocchio and reranking is the Part 4 feature.

## 3. Requirement by requirement

### R1: web chat and UI controls (1p)
- Chat with a playlist panel under it: numbered tracks, a **Remove** button per track, a **Clear**
  button and an **Add** box. The buttons send the same commands the chat understands (`/remove <id>`,
  `/clear`, `/add <text>`), so there is one code path for chat and UI.
- **Why:** one path means no state can drift apart. The server is the single source of truth; every
  reply carries the playlist (`PLAYLIST_STATE` act) and the panel simply renders it.
- Buttons for choices (R4/R6) come from an `OPTIONS` act; clicking one sends the label as text, so
  the pick goes through the same code as typing.

### R2: add, remove, view, clear in natural language, with a database (3p)
- **Database.** `import_tracks.py` loads the catalog into SQLite: a `tracks` table with
  pre-normalised title/artist columns plus release date, duration and tags, and a `track_artists`
  table (one row per artist and track). Only ids from this table can enter the playlist; the
  playlist stores track ids, never free text. LLM output (R8) is also checked against the database.
- **Why SQLite:** it needs no server, loads in seconds, and an indexed lookup on the normalised title
  is instant.
- **Understanding.** The LLM turns "add", "remove", "show", "clear" in any wording into an action.
  The regexes (`_ADD_RE`, `_REMOVE_RE`, `_VIEW_RE`, `_CLEAR_RE`) cover the same phrasings when the
  LLM is not available.
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
- **Removal** matches words of the request against the title and artist of the playlist songs, so
  "remove nirvana" works. If several playlist songs match it asks which one instead of guessing.
- **Duplicates** are never added twice ("already in your playlist").

### R3: help in natural language (1p)
- `capabilities.py` is a registry: for each feature a title, summary, keywords and an example text.
  The LLM's `help` action (or the keyword check in the fallback) replies with the whole list, or only
  the feature the user asked about.
- **Why a registry:** the overview, the focused answers and the "available" flag all come from the
  same data, so adding a feature means adding one entry, and help can never advertise something that
  is not implemented.

### R4: disambiguation with intelligent ranking (1p)
- When several different songs match ("Love", "Yellow", "Iris"), the agent shows a numbered list with
  buttons. Duplicates of the same song (same title and artist on different albums) are collapsed.
- **Ranking** (`_rank_for_playlist`): `popularity/100 + 1.0 if the artist is already on the playlist
  + 0.5 if the album is`. A popular song, or one that fits what you already chose, comes first.
- **State machine.** The list is stored as `_pending` together with a callback `then(track, note)`
  and the text for "cancel". The user can answer with a number, an ordinal, the artist, the button,
  or "cancel". A new request closes the list and is executed normally, so the user is never trapped.
- **Why a callback:** the same list serves four jobs - add a song, answer a question about it (R5),
  remove it (R2), and add recommendations (R6). Only `then` differs.

### R5: questions about tracks and artists (1p, needs 4 types)
- The LLM's `ask` action carries a kind and a subject; `_ASK_KINDS` maps the kind to the answer code.
  **Songs:** artist, album, release date, length, popularity, tags/genre, summary. **Artists:**
  songs, albums, how many songs, most popular song, tags/style, summary. Tags and summary work for
  both, so there are 11 distinct kinds, well over the 4 required. `questions.py` (about 40 regex
  patterns) answers the same questions as the fallback.
- Examples: `who sings Yellow?`, `when was Yellow released?`, `how long is the last one?`,
  `what genre is Coldplay?`, `how many songs does Coldplay have?`, `tell me about Nirvana`.
- **Context:** "it", "the second one", "the last one" are resolved from the numbered playlist and the
  last song in the state; the code then answers from the database.
- **Artist matching** uses an artist table with a leading "the" removed ("The Beatles" = "Beatles")
  and `difflib` for typos; the answer says what it understood.
- **Why the answers still come from code:** every fact (year, length, popularity, tags) is read from
  the database, so nothing is invented; the LLM only decides *which* question was asked.
- **Tags.** See section 4.

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
- With the open list in its state, the LLM's `select` action returns positions, `mode` all/none, and
  an `except` list. This covers numbers (`1 and 3`), position words (`the first and the last`),
  quantities (`the first two`), all (`add them all`), exclusions (`all but the second`,
  `everything except the Linkin Park one`), by artist (`the ones by Coldplay`) and none
  (`nah, skip those`). Code checks the result: a number outside the list is never guessed.
- `selection.py` (`parse_selection`) does the same without the LLM and is the fallback: numbers,
  ordinals, quantities, ranges (`2-4`), all, exclusions and names (`the Nirvana ones`).
- In a question list ("when was Iris released?") picking several answers "that is about one song"
  instead of adding anything. Several songs are added by `_add_many`, which skips songs already on
  the playlist and says so.

### R8: a whole playlist from a description (1p)
- `make me a playlist for the gym`, `I need something for a long road trip`, `/generate ...`.
- **Steps.** (1) One LLM call turns the request into JSON: search keywords and a duration in minutes
  (the LLM picks a sensible length itself: gym 45-75 min, studying 60-120 min, road trip 90-180 min;
  we clamp to 15-120 min). (2) Candidates come from two retrievers fused with reciprocal rank
  fusion: our Part 1 BM25 retriever over names and tags, and a dense retriever (Qwen3 embedding of
  the keywords against the precomputed track-metadata embeddings). (3) Songs are added until the
  target duration is reached, skipping tracks with missing or implausible durations, at most 3 per
  artist and at most 40 songs. (4) Ids are verified against the database.
- **Length without asking:** the user is never asked for a number of songs; the duration target
  drives it, and the reply states the result ("13 songs, approximately 62 minutes").
- **Why fusion:** on 1,500 training requests, BM25 alone reached nDCG@20 0.0655 and BM25 + metadata
  embeddings 0.0829. BM25 finds the literal words, the embeddings find the meaning.
- **Why the prompt forbids occasion words:** BM25 matched "gym" in the band Gym Class Heroes. The
  prompt tells the LLM to describe the music (genre, mood, tempo) and not the place or activity;
  themes the user names ("sad", "love", "rain") and artist names are kept.
- **Safe overwrite:** a generated playlist replaces the current one, so with a non-empty playlist the
  agent first asks "replaces your current playlist (N songs). Say yes or no" and does nothing until
  the answer; the LLM reads the answer ("yeah go for it") from the pending question in its state.
- **Speed:** the BM25 index, the catalog and the embedding model are loaded at server start
  (about 30-40 s once). A request then takes about 2 s, mostly the LLM. If the embeddings cannot
  load, generation falls back to BM25 only.

## 4. Tags: judged by the LLM, not by a hand-written list

The catalog's tags are free text typed by Last.fm users: 163,829 distinct tags, 99,386 of them used
by only one track. They contain genres and moods but also opinions ("favorites", "awesome"), ratings
("10 of 10 stars"), ownership ("albums i own"), years, radio stations and typos.

- **First attempt (replaced):** a hand-written stop list of 13 words. It missed almost everything.
- **Now:** `scripts/label_tags.py` asks the LLM once, offline, about each tag used by at least 10
  tracks (14,768 tags, about 2 minutes): does the tag describe the music itself (genre, mood,
  instruments, vocals, tempo, era, language, region, a situation the music suits) or is it noise?
  The answer is stored in `data/tag_labels.json` (5,331 tags kept, 36%). The agent only reads that
  file. Rarer tags count as noise by frequency alone, and a tag equal to the song's own title, artist
  or album is dropped. There are no stop lists or filter patterns in the code.
- **Where it is used:** song and artist tag answers (R5), the reasons shown with recommendations,
  and the tag-similarity feature of the recommender (R6).
- **Effect on quality:** recommendation nDCG@20 on the 1,337 training turns was 0.2275 with the
  judged tags against 0.2304 with the old list - the same within noise - with a vocabulary of 5,331
  tags instead of 164,000.
- **Where it is not used:** R8's search keeps the raw tags. On 400 real requests BM25 over the
  judged tags scored 0.0445 against 0.0613 over raw tags (fused with embeddings 0.0674 against
  0.0727): raw tags also hold artist names and other cues that help when a user names an artist.
- **Why offline:** no LLM call at run time, the result is reproducible and can be read and corrected.

## 5. Cross-cutting design choices

| Choice | Why |
|---|---|
| LLM chooses the action, code does the work | New wordings and context are understood, but only database songs and database facts can reach the user |
| Rules as the fallback | No LLM, a slow model or a bad reply never breaks the chat; the demo keeps working |
| Exact inputs (`/commands`, button clicks) skip the LLM | Deterministic and fast where the input is already exact |
| State summary in the prompt | "the second one", "it" and "yes" can only be understood with the playlist, last song, open list and pending question |
| Playlist stores track ids only | Nothing outside the catalog can ever be added; the UI and the agent share one truth |
| One "pick from a list" mechanism with a `then` callback | R4, R5, R6, R2-remove all reuse it, so the UI buttons and the typed answers behave identically |
| Heavy objects built once per process, warmed in `__main__` | DialogueKit creates a new agent per user; indexes must not be rebuilt per connection |
| Tags judged once offline by the LLM | Less hard-coded knowledge, no run-time cost |
| Replies escape user text (`html.escape`) | The UI renders replies as HTML |
| Verified by tests and live runs | 478 automated tests plus a scripted conversation and a 43-message router test against the real model |

## 6. Evidence

- **Part 1 retriever:** final score 0.2857 on the test split against the target 0.22; the same
  Part 1 feature ideas (artist, album, last artist, popularity) drive R6, and the retriever drives R8.
- **Router:** 43/43 correct on the labelled messages, median 0.5 s, slowest 1.6 s per call.
- **End-to-end run (real LLM, real embeddings, LLM-first):** every turn under 3 s; a generated
  playlist took about 2 s, most other turns 0.2-1.2 s including the router call, and one question took
  2.8 s when the model answered slowly. The server warm-up takes about 30-40 s once.
- **Tests:** `pytest` runs 478 tests covering every requirement; the catalog tests use the real
  database, and the router tests use a scripted fake LLM (every action, the fallback paths, slash
  commands and button clicks that skip the LLM).

## 7. Known limits (say them before the examiner finds them)

- With the LLM on, each message costs one extra model call (about 0.5-1 s); when the model server is
  down the rule-based parsers answer, which understand fewer wordings.
- The LLM can misjudge an ambiguous message (for example "more of that" can mean recommend or
  feedback). Code validates the action and falls back when fields are missing, but not every
  misjudgement can be caught.
- `recommend` does not use the hint yet, and `feedback` is only recorded (Part 4).
- Search keywords can still match song titles literally ("High Fidelity" for "high energy").
- The tag labels are the LLM's judgement; a few borderline tags ("fond memories") were kept.
- Mood and activity playlists were judged by reading the results, not with a quality metric.
- The first server start downloads about 1.2 GB (embedding model) and 0.8 GB (track embeddings).

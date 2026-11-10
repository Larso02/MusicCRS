# MusicCRS implementation log: what we built and why

Last updated: 2026-10-29. See `full_overview.md` for the complete story in order (every approach tried, with numbers,
and why). This file is the working record. Keep it current: when something is added, changed or
rejected, add a line to section 3 or 4 with the reason and the evidence, and update the tables. More detail
lives in `part2_summary.md` (requirements R1-R8), `part4_results.md` (Part 4 numbers),
`part4_proposal_draft.md` (the proposal text and the lecturer feedback) and `part1_experiment_log.md`.

## 1. What exists

### Code (`musiccrs/`)
| File | What it does | Why it exists |
|---|---|---|
| `musiccrs.py` | the agent: receives each message, routes it, carries it out | one place for the dialogue state (playlist, open list, profile) |
| `router.py` | the LLM reads a message plus the state and returns one JSON action | the lecturers asked for more LLM and less hard-coded wording |
| `track_database.py` | SQLite queries, text normalisation, tag helpers | only catalog songs may ever reach the playlist |
| `fuzzy.py` | typo-tolerant track search (trigram BM25 + difflib) | R2 loose references |
| `questions.py` | rule-based recognition of questions about songs/artists | R5 fallback when the LLM is off |
| `selection.py` | rule-based "the first two", "all except 2" | R7 fallback |
| `capabilities.py` | the feature list behind the help text | R3; help can never advertise a missing feature |
| `search_api.py` | `GET /search`: search the whole catalog while typing (with a typo fallback) | the UI's search box must answer on every keystroke without adding chat messages |
| `recommender.py` | "recommend" from the playlist (Part 1 features + tags + year) | R6 |
| `playlist_generator.py` | a whole playlist from a description (LLM keywords, BM25 + embeddings, duration) | R8 |
| `user_profile.py` | what the user liked and rejected during the chat | Part 4 session preference profile |
| `refine.py` | Rocchio query modification, neural reranking, MMR, explanations | Part 4 |
| `neural.py` | similarity scoring in embedding spaces; style prototypes | Part 4 neural reranker |
| `reranker_model.py` | the two-tower network (model, save/load, held-out session rule) | Part 4 neural reranker |

### Scripts and data
| File | What it does |
|---|---|
| `scripts/import_tracks.py` | builds `data/tracks.db` (tracks, artists, release date, duration, tags) |
| `scripts/label_tags.py` -> `data/tag_labels.json` | the LLM judges which tags describe music (once, offline) |
| `scripts/train_reranker.py` -> `data/reranker.pt` | trains the two-tower network (about 7 minutes) |
| `scripts/eval_router.py` | 43 labelled messages against the real LLM (accuracy, latency) |
| `scripts/eval_feedback.py` | three sets of 30 feedback sentences against the real LLM |
| `scripts/transcribe.py` | local Whisper transcription of the Part 3 interview recordings |
| `experiments/00-09` | Part 1 and feasibility experiments; `10_refine_eval.py` is the Part 4 evaluation |
| `frontend/.../ChatBox.tsx` | the chat (left column); `SidePanel/SidePanel.tsx` is the right column: taste chips with a Forget button, the playlist with a filter box and Remove/Clear, and "Find songs" (search all songs, Add button) |

## 2. How a message is handled now
1. `/commands` (the UI buttons) and clicks on option buttons skip the LLM: they are exact.
2. The LLM (`router.py`) returns one action: add, remove, view, clear, help, ask, recommend, select, generate,
   confirm, feedback, preferences, forget, why, chat or quit. Its input is the message plus a state summary
   (numbered playlist, last song, open list, last recommendations, pending question).
3. Our code carries the action out with the existing, tested functions. The LLM never supplies songs or facts.
4. If there is no LLM, the call takes more than 4 s, or the reply is unusable, the regex parsers answer.
5. Recommendations: with no feedback the R6 recommender is used; once the profile has content, or variety is
   requested, the Part 4 pipeline runs (Rocchio -> BM25 candidates -> neural reranker -> constraint -> MMR).

## 3. Decisions and why

### Part 1 retriever
- BM25 over the chat text + playlist features (same artist/album/last artist/popularity) + a usage penalty
  across all turns. Final score 0.2857 against the target 0.22 (baseline 0.1446). The usage penalty removes
  "hub" tracks that are recommended everywhere and almost never correct, so it helps nDCG and diversity.
  Details and the experiments are in `part1_experiment_log.md`. Tuned on train only.

### Catalog and playlist (R1, R2)
- SQLite with normalised title/artist columns and a `track_artists` table: an exact indexed lookup, no server.
- The playlist stores **track ids only**: nothing outside the catalog can be added, and the UI and agent
  share one truth (every reply carries the playlist; the panel just renders it).
- UI buttons send the same commands as typed chat (`/remove <id>`, `/clear`, `/add <text>`): one code path.

### Loose references (R2)
- `_lookup` is the single "find one track" routine: id -> exact title (+ artist) -> several songs go to the
  pick-list -> typo-tolerant search (trigram BM25 + difflib; >= 0.9 accepted with a note, 0.6-0.9 asks).

### Pick-lists (R4, R7)
- One mechanism with a `then(track, note)` callback serves add, answer-a-question, remove and recommendations.
- Ranking of choices: popularity/100 + 1.0 artist on the playlist + 0.5 album on the playlist.
- A new request closes an open list, so the user is never trapped.

### Questions (R5)
- 11 kinds (4 required). Answers are read from the database. The LLM only decides which question was asked,
  so nothing is invented. `questions.py` (rules) answers the same questions without the LLM.

### Recommendations (R6)
- Score = same artist 1.0 + same album 0.5 + last artist 1.0 + popularity 1.0 + tag similarity 1.0 + year 0.5.
  nDCG@20 0.2205 (without tags/year) -> 0.2317 on 1,337 training turns. Max 2 songs per artist and 2 by
  artists already on the playlist, so the list is not "more of the same". BM25 over the playlist added nothing.

### Selection (R7)
- Rule parser `selection.py` (numbers, ordinals, ranges, quantities, all, exclusions, names) plus the LLM's
  `select` action. A number outside the list is never guessed.

### Playlist generation (R8)
- One LLM call gives search keywords and a duration (clamped to 15-120 min); BM25 and Qwen3 embeddings are
  fused with reciprocal rank fusion (nDCG@20 0.0655 BM25 alone, 0.0829 fused); songs are added until the
  duration is reached (<= 3 per artist, <= 40 songs); ids are verified against the database.
- The prompt keeps the occasion out of the keywords ("gym" matched the band Gym Class Heroes) while keeping
  themes the user names and artist names.
- Replacing a non-empty playlist asks first.

### LLM-first understanding
- Feedback from the Part 2 presentation: use more LLM, the regexes are too hard-coded. The router is now first;
  regexes are the fallback. 43/43 correct on the labelled messages, median 0.5 s. One call per message keeps us
  far below the limit of 5-10 LLM calls per turn.
- Prompt fixes that mattered: number the playlist; write songs as "Title by Artist" (otherwise the artist was
  dropped); `variety` only when the user explicitly asks for a different mix; opinions about listed songs are
  feedback, not a select.

### Tags
- The tags are free Last.fm user text (163,829 distinct, 99,386 used once). A 13-word stop list was replaced by
  an **LLM judgement made once offline** (`label_tags.py`): 14,768 tags judged, 5,331 kept as describing the
  music. No stop lists or filter patterns remain in the code. The recommender is as accurate (0.2275 against
  0.2304, noise) with a vocabulary 30 times smaller.
- R8's search keeps the **raw** tags: BM25 over the judged tags scored 0.0445 against 0.0613 (raw), because raw
  tags also carry artist names and other cues.

### Part 4: feedback-aware refinement
- **Profile** (`user_profile.py`): liked/rejected tracks, artists, styles, wishes; newest opinion wins.
  Implicit feedback: adding a suggested song = like, removing one = dislike.
- **Feedback understanding**: the router's `feedback` action (positions, artists, styles, a wish, variety).
- **Rocchio** (query modification before retrieval), beta tuned to 1.2, negatives clipped (BM25 cannot score
  negative weights). BM25 only retrieves the top 1000 candidates.
- **Neural reranker** (lecturer feedback, see section 6). A two-tower network trained on 90% of the TalkPlay
  training sessions over the Qwen3 metadata and BPR collaborative-filtering embeddings. Liked styles are
  represented by the mean embedding of the tracks with the tag (a "style prototype") in the trained space;
  wishes that match no tag are embedded as text; rejected songs/styles are penalties; explicitly rejected
  artists/styles are removed as a constraint. No BM25 and no tag counting in the reranker.
- **MMR** in the trained embedding space. "More varied" uses lambda 0.3 over a pool of 200; default 0.8.
- **Explanations** come from the stored score components; a before/after list difference is shown; the LLM only
  words the opening sentence.
- **Transparency** (added last): the UI shows taste chips with a Forget button; the chat answers "what do you
  know about my taste?", "forget my preferences" and "why did you suggest X?" from stored facts.

### Web UI: two columns, playlist filter and catalog search (added after Part 4)
- Why: adding a song used to mean typing its name into a box or into the chat, with no way to look around the
  catalog, and the playlist panel sat under the chat where it was hard to see. Now the chat is the left column and
  the right column holds the taste chips, the playlist and a song search, so the playlist is always in view.
- **Playlist filter**: client-side, every word must occur in title, artist or album; songs keep their real number
  so "remove number 7" still matches what the user sees. Clear asks for confirmation when there is more than one song.
- **Catalog search**: `GET /search?q=...` on the same Flask app as the socket. Why a plain route and not the chat
  socket: a search box needs an answer on every keystroke, and sending each one as a chat message would fill the
  conversation. The UI waits 250 ms after the last key and drops answers to outdated queries.
- **How it searches** (`TrackDatabase.search_catalog`): every word must occur in the title or artist (so
  "yellow coldplay" and "coldplay yellow" both work, and partial words work while typing); exact title first,
  then titles that start with the text, then by popularity. Case, accents and punctuation do not matter.
- **Typos**: the fuzzy index is used only when nothing matches exactly ("stnad by me" -> Stand By Me). In the first
  version close spellings were also added after real matches and padded "yellow coldplay" with songs that do not
  contain "yellow"; fuzzy guesses are now offered only instead of an empty result, with a minimum similarity of 0.7.
- **Adding from the UI** sends `/add <track id>`, the same exact command as before, so a click can never pick the
  wrong song. A song already on the playlist shows "In playlist" instead of an Add button; with no result the user
  can still send the text to the chat's fuzzy add ("Try adding it by name").
- Narrow screens and the floating widget get one column (chat, then the lists below).
- Not changed: the chat itself, the router and all recommendation logic.

### Part 3 interviews
- Recorded on a phone, transcribed locally with Whisper (`transcribe.py`, no cloud upload), names replaced by
  `[NAME]`. Forcing the language (English) fixed a poor first pass that had detected Norwegian. Recordings and
  transcripts are kept out of git.

## 4. Things we tried and rejected (so nobody repeats them)
| Idea | Result | Why rejected |
|---|---|---|
| Tag-only BM25 for R8 | 0.054 vs 0.073 nDCG | real requests name artists and songs |
| BM25 over the judged tags for R8 | 0.0445 vs 0.0613 | loses artist-name and other cues in raw tags |
| BM25 over the playlist metadata for R6 | no gain | the other features already cover it |
| Embedding centroids alone for R6 | 0.146 vs 0.219 | worse than artist/album/popularity |
| BM25 + tag-overlap reranker for Part 4 (v3 proposal) | worked (+14% with Rocchio) | lecturer: BM25 is not reranking, tag reranking makes no sense |
| Zero-shot cosine reranker | 0.19 vs 0.22 (small test) | plain cosine to the playlist hurt; the network must be trained |
| Request text as a reranking anchor | hurt (noisy chat text) | text is used only for wishes that match no tag |
| Embedding the style word with the text encoder | nDCG 0.234 vs 0.277 | style prototypes (mean embedding of tracks with the tag) work |
| Mixing raw cosines into the trained network | 0.249-0.260 | no better than the network alone (0.2594) |
| Soft penalty for rejections | cut only 18-31% | explicit rejections are now a constraint (100%) |
| Meeting-tool transcripts (Teams/Zoom) | needed subscriptions / no speaker split | local Whisper is free and private |

## 5. Evaluation record
- **Router** (`eval_router`): 43/43, median 0.5 s, slowest 1.2 s.
- **Feedback parsing** (`eval_feedback`, 30 sentences each): tuned set 28/30 (93%); a second set seen once and
  used for one principled fix 27/30 (90%); a **fresh set run once, nothing changed afterwards: 28/30 (93%)**.
  Rules disclosed: a `select` counts as "liked these" (adding a suggested song registers a like) and a
  `recommend` hint counts as a wish; both were added after the first look at the results and apply to all
  sentences. All sentences were written by the author of the prompt: add sentences from someone else for a
  stronger test. Misses seen: "just more of what you gave me" (variety "less"), "give me upbeat summer songs"
  (read as generate a playlist), "I'm in the mood for jazz" (generate), one 4 s timeout.
- **Part 4 ranking** (`experiments/10_refine_eval.py`, 1,500 held-out turns): see `part4_results.md`.
  nDCG@20: Part 1 0.2251; + Rocchio 0.2580; + neural reranker 0.2372; both 0.2594. Rejecting a tag: share of
  the top 5 0.85 -> 0.00 with the constraint (0.58 without). Asking for a style: share 0.13 -> 0.25 with nDCG
  0.247 -> 0.305. MMR lambda 0.4: diversity 0.265 -> 0.302.
- **Latency**: router 0.5 s; a feedback turn (two LLM calls) 1.0-2.9 s; a generated playlist about 2 s; server
  start 40-60 s.
- **Tests**: 587 automated Python tests (section 9) and 16 frontend tests.

## 6. Feedback timeline and our response
| When | Feedback | What we did |
|---|---|---|
| Part 1 | target score 0.22 | BM25 + features + usage penalty, 0.2857 |
| Part 4 proposal v1 -> v2 | Rocchio ambitious; parsing = 2p; add a 1p task; latency a general condition; explanations must add value | thresholds softened, implicit feedback added, explanations computed from score components |
| v2 -> v3 | "Rocchio is not reranking"; implement reranking with BM25 and tags plus Rocchio for 4p | Rocchio described as query modification; added a separate reranker |
| v3 -> v4 | "BM25 is not used for reranking; tag reranking does not make sense; reranking needs a neural method; tags may be added" | trained two-tower neural reranker; tags only as style prototypes; BM25 only retrieves |
| Part 2 presentation | use more LLM; regexes too hard-coded; manual tag filtering | LLM-first router with regex fallback; tags judged by the LLM |

## 7. Known limits and open items
- Rocchio gives most of the accuracy gain; the neural reranker replaces BM25 scoring at equal accuracy (0.2594
  vs 0.2580) and carries the feedback (styles, wishes, penalties). Raw cosines are competitive (0.2409).
- Only the constraint fully removes a rejected attribute; the soft neural penalty cuts 18-31%.
- Sections B and C of the Part 4 evaluation use constructed scenarios, not real users; nDCG rises in scenario B
  by construction.
- With the LLM on, each message costs a model call; when the model server is down the regex parsers answer and
  understand fewer wordings.
- `recommend`'s hint becomes a stated wish; all feedback is session-level (nothing is stored across chats).
- Tag labels are the LLM's judgement; a few borderline tags were kept.
- Song titles are placed in the router prompt: actions are validated and only the database supplies songs, but a
  hostile title could still steer the wording of a reply.
- The first run on a new machine needs the embedding caches (about 0.8 GB) and `data/reranker.pt` (committed).
- No type checker (Pyright) was run in our environment; Pylance may show warnings.

## 8. Reproduce
```
python -m scripts.import_tracks                  # build data/tracks.db
python -m scripts.label_tags                     # (optional) re-judge the tags -> data/tag_labels.json
python -m scripts.train_reranker                 # (optional) retrain -> data/reranker.pt, about 7 min
python experiments/10_refine_eval.py --turns 1500 --scenarios 400 --tune     # Part 4 numbers, about 1 hour
python -m scripts.eval_router                    # router accuracy (needs MY_OLLAMA_API_KEY)
python -m scripts.eval_feedback --set fresh      # feedback parsing; also --set tuned / heldout
python -m pytest -q                              # 587 tests
cd frontend && npx tsc --noEmit -p . && CI=true npx react-scripts test --watchAll=false   # frontend: types and 16 tests
python musiccrs/musiccrs.py                      # backend; frontend: cd frontend && npm start
```

## 9. Test inventory (587 Python tests + 16 frontend tests)
| File | Tests | Covers |
|---|---|---|
| test_agent.py | 17 | basic agent behaviour, quit, playlist state |
| test_track_database.py | 16 | catalog queries |
| test_fuzzy.py | 15 | typo-tolerant search |
| test_help.py | 21 | help text |
| test_choices.py | 20 | pick-lists (R4) |
| test_questions.py | 77 | questions about songs and artists (R5) |
| test_recommend.py | 29 | recommendations (R6) |
| test_selection.py | 85 | selection (R7) |
| test_generate.py | 46 | playlist generation and the replace question (R8) |
| test_part2.py | 73 | view/clear phrasings, removal, the assignment's selection examples |
| test_tags.py | 32 | LLM-judged tags and the labelling script |
| test_router.py | 47 | every router action, fallbacks, slash commands, button clicks |
| test_refine.py | 35 | profile, Rocchio, constraint, MMR, explanations, feedback flow |
| test_neural.py | 19 | embeddings, style prototypes, the trained network, held-out rule |
| test_transparency.py | 22 | taste chips, preferences, forget, why |
| test_search.py | 26 | catalog search: ranking, words in any order, typos, limits, SQL safety, speed, the HTTP route |
| frontend/src/tests/SidePanel.test.tsx | 16 | playlist filter, Remove/Clear, search with delay and errors, Add, taste chips |
| test_eval_feedback.py | 7 | the feedback-evaluation checker and the integrity of its three sets |

Frontend test note: `npm test` first failed with a Babel version error that was already in the project (the installed
`@babel/core` 7.20 is older than another package needs). The tests were run after `npm install --no-save @babel/core@7.24.0`,
which changes nothing in `package.json` or the lock file; on another machine, run the same command if `npm test` fails
with "Requires Babel ^7.22.0".

# MusicCRS: the full overview - everything we implemented, everything we tried, and why

Last updated: 2026-10-29. This is the one file to read to understand the whole project. It tells the story in
order (Part 1 to Part 4), and for every area lists what was built, what alternatives were tried, the numbers that
decided between them, and the reason for the choice. Shorter, more technical companions:

- `implementation_log.md` - system map, decision table and test inventory (keep this one current on every change)
- `part1_experiment_log.md` - the full Part 1 experiment notebook
- `part2_summary.md` - the eight Part 2 requirements, how each is met
- `part4_results.md` and `part4_proposal_draft.md` - Part 4 results, and the proposal text with the lecturer feedback
- `demo_script.md` - what to say and click in the demo

## Contents
1. The project in one page
2. Part 1: retrieval (item ranking)
3. Part 2: the chat system, requirement by requirement
4. The LLM-first redesign
5. Tags: from a stop list to an LLM judgement (and 5b, the web UI)
6. Part 3: the interviews
7. Part 4: feedback-aware refinement (all iterations)
8. How we evaluated, and how we kept the evaluation honest
9. Bugs and corrections that taught us something
10. Everything we tried that did not make it
11. Lessons
12. Limits and open items
13. Who did what, and how to reproduce

---

## 1. The project in one page

MusicCRS is a conversational music recommender: a web chat where the user builds a playlist from a catalog of
47,071 TalkPlay tracks. It grew in four steps.

| Part | Goal | Outcome |
|---|---|---|
| 1 | rank the catalog for a conversation turn (batch retrieval) | score 0.2857 against the target 0.22 and the BM25 baseline 0.1446 |
| 2 | a chat system for playlists, questions, recommendations, selection, generation (R1-R8) | all eight requirements; 587 automated Python tests and 16 frontend tests |
| 3 | evaluate other groups' systems by interview | two interviews recorded and transcribed locally |
| 4 | advanced feature: feedback-aware refinement | profile + Rocchio + trained neural reranker + MMR + explanations |

Design principles that run through everything:
1. **The database decides what exists.** The playlist stores track ids only; the LLM never supplies a song or a
   fact. This is why we can use an LLM freely without the system inventing music.
2. **Measure first, then decide.** Every non-obvious choice below has a number next to it, and most of them came
   from an experiment that could have gone the other way.
3. **Keep a fallback.** The LLM can be slow, absent or wrong; the rule-based parsers still work.
4. **Be honest about the evaluation.** Tune on train, check on held-out data, disclose what was looked at twice.

---

## 2. Part 1: retrieval (item ranking)

Task: given a conversation, rank the catalog and return 20 tracks. Score = 0.8 * nDCG@20 + 0.2 * catalog diversity
(unique recommended tracks / 47,071). Rule we followed: tune on train, report on test.

### What we measured first (train, 121,592 turns)
| Finding | Consequence |
|---|---|
| the target was already played earlier in the session in **0.0%** of turns | always remove played tracks |
| 43,597 of 47,071 tracks are a target somewhere | no small "hot set"; pure popularity cannot work |
| the target's **artist** was already in the session in **60.7%** of turns (album 41.6%) | artist/album continuity is a strong signal |
| targets are slightly more popular (47.8 vs 35.8) | a small popularity prior may help |

### What we tried, in order
| # | Idea | Result (train sample unless noted) | Decision |
|---|---|---|---|
| 1 | BM25 baseline (history as query) | test final 0.1446 | reproduced the grader exactly |
| 2 | current message only as query | 0.053-0.080 | rejected: the history matters far more |
| 3 | add `tag_list` as a BM25 field | no gain once track metadata is in the query | rejected |
| 4 | **drop already-played tracks** | 0.1076 -> 0.1552 (+44%); test final 0.1663 | kept (method A, first submission: 29% on QuickFeed) |
| 5 | weighted rescoring: BM25 + artist + album + last artist + popularity | 0.1412 -> 0.1719 (+22%), +20% on an independent sample | kept; weights tuned on an 8,000-turn dev set, held-out half 0.2095 vs 0.2155 (no overfitting) |
| 6 | add the current message's BM25 to the score | worse (0.1553 -> 0.1417) | dropped (w2 = 0) |
| 7 | **usage penalty** `score - lam * log(1 + times recommended)` over all turns | nDCG 0.2125 -> 0.2490, diversity 0.43 -> 0.75 (dev set) | kept, lam = 1.0 |
| 8 | protect the top-p ranks from the penalty | worse for every p | rejected (keeps the hub tracks on top) |
| 9 | penalty lam = 2.0 | nDCG drops | rejected |
| 10 | candidate pool K = 300 vs 1000 | final 0.2717 vs **0.2857** | K = 1000 |
| 11 | dense retrieval / fusion / learned reranker | not needed: the target was already beaten | postponed (these ideas came back in Parts 2 and 4) |

Why the usage penalty works (the "hubness" finding): 70 tracks appear in 100 or more of the 8,000 lists and are
correct in only **0.04%** of their slots, against 3.7% for tracks recommended at most four times. Repeating the same
popular tracks wastes slots, so penalising repeats raises nDCG *and* diversity. The method is transductive (it uses
all test queries together), which fits a batch task but not a live chat; the report says so.

Final pipeline (`retrieval/run_final.py`): history query -> BM25 over the catalog -> score with artist, album,
last-artist and popularity -> drop played tracks -> top 1000 -> usage-penalty pass over all turns.

| stage (test split) | final score |
|---|---|
| BM25 baseline | 0.1446 |
| + drop played tracks | 0.1663 |
| + artist/album/last-artist/popularity | 0.2101 |
| + usage penalty, K = 300 | 0.2717 |
| + usage penalty, K = 1000 (submitted) | **0.2857** (nDCG@20 0.1755, diversity 0.7263) |

Honesty notes: weights were chosen on train-derived data; K (two values) and a few values of lam were compared on
test. The local evaluator matched the grader's numbers exactly, so local tuning is trustworthy.

---

## 3. Part 2: the chat system, requirement by requirement

Rules of the assignment: responses within 3-5 s, 5-10 LLM calls per turn at most, only catalog songs may be
recommended or added, loose references to songs and artists must work.

### R1 - web chat and UI controls
- Built: chat, a playlist panel with a Remove button per song, a Clear button and an Add box.
- Why this way: the buttons send the same commands the chat understands (`/remove <id>`, `/clear`, `/add <text>`),
  so there is **one code path** for chat and UI. Every reply carries the playlist, so the panel and the server
  can never disagree.

### R2 - add, remove, view, clear with a database
- Database: SQLite with normalised title/artist columns and a `track_artists` table. Alternatives were not needed:
  no server, loads in seconds, an indexed title lookup is instant.
- Loose references (`_lookup`): track id -> exact title (+ "by artist") -> several songs open a pick-list ->
  typo-tolerant search -> "I couldn't find". The typo search uses a character-trigram BM25 index for candidates and
  `difflib` to order them; a score >= 0.9 is accepted with a note ("I read 'stnad by me' as 'Stand By Me'"),
  0.6-0.9 becomes "Did you mean?".
- Why trigrams + difflib: trigram BM25 finds candidates among 47k titles even with swapped letters; difflib then
  compares whole strings.
- Removal matches words against title and artist of the playlist songs; several matches open a pick-list instead
  of guessing. Duplicates are never added twice.

### R3 - help in natural language
- A registry (`capabilities.py`) with title, summary, keywords and example text per feature. The overview and
  the focused answers come from the same data, so help cannot advertise a feature that does not exist.

### R4 - disambiguation with intelligent ranking
- A numbered list with buttons; duplicates of the same song collapsed; ranked by
  popularity/100 + 1.0 (artist already on the playlist) + 0.5 (album already on the playlist).
- Implementation: a small state machine whose pending choice carries a **callback** `then(track, note)`. The same
  list therefore serves four jobs (add a song, answer a question about it, remove it, add recommendations); only
  the callback differs. A new request closes the list, so the user is never trapped.

### R5 - questions about songs and artists (4 types required)
- 11 kinds: song artist/album/year/length/popularity/tags/summary; artist songs/albums/count/top/tags/summary.
- Every answer is read from the database; "it" and "the last one" resolve from the state. Popularity is shown as a
  score and as a percentile.
- Why not let the LLM answer: facts must be exact, and an LLM can invent them. The LLM only decides *which*
  question was asked (since the LLM-first redesign); the rule table (`questions.py`) is the fallback.

### R6 - recommend 3-5 songs related to the playlist
- Score = same artist 1.0 + same album 0.5 + same artist as the last song 1.0 + popularity 1.0 + tag similarity
  1.0 + release year 0.5 (year decays with distance). Each song comes with a reason.
- What we tested (1,337 training turns "predict the next song from the earlier ones", weights chosen on one
  half, checked on the other):

| Variant | nDCG@20 |
|---|---|
| artist + album + popularity (Part 1 idea) | 0.2205 |
| + tag similarity and release year | **0.2317** (kept) |
| + BM25 over the playlist metadata | 0.2372 on the check half, 0.2317 without; no clear gain, dropped |
| embedding centroids alone: metadata 0.1457, collaborative filtering 0.1150, attributes 0.0676, audio 0.0348 | much worse than the metadata features |

- Variety: at most 2 songs per artist and at most 2 by artists already on the playlist (without it 4 of 5 songs
  were by the same two artists). "Recommend more" never repeats earlier suggestions; the scan window had to be
  widened (from the top 200 to the top 2,500 candidates) because after a few rounds the list shrank to 2 songs.

### R7 - selecting songs in natural language (3 formulations required)
- A pure parser (`selection.py`): numbers, lists, ordinals, ranges (`2-4`), quantities (`the first two`),
  all, exclusions (`all except 2`, `add them except the one by Metallica`), names (`the Coldplay ones`), none.
  A number outside the list is never guessed; a bare artist name that matches two songs asks again.
- "one" is not treated as the number 1 ("the second one" must not also select 1).
- In a question list, picking several songs answers "that is about one song" instead of adding anything.

### R8 - a whole playlist from a description
- One LLM call -> search keywords + a duration (clamped 15-120 min). Candidates: BM25 over names and tags fused
  (reciprocal rank fusion) with the Qwen3 metadata embeddings; songs are added until the duration is reached
  (<= 3 per artist, <= 40 songs); every id is checked against the database. The length is chosen by the system;
  the user is never asked for a number.
- Evidence for the fusion (1,500 requests): BM25 0.0655, embeddings alone 0.0645, **fused 0.0829**.
- The "gym" problem: BM25 matched the word "gym" in the band *Gym Class Heroes*. Changing which fields BM25
  searches made things worse (tags only 0.0315 vs 0.0613; names only 0.0414), so we fixed the **prompt**: the
  LLM describes the music (genre, mood, tempo) and not the place or activity, keeps themes the user names and
  keeps artist names. A first wording that forbade too much dropped "love" from "sad love songs"; the final
  wording keeps themes.
- Weighting the embeddings 2:1 over BM25 made results more literal; 1:1 was kept.
- Replacing a non-empty playlist asks first. The first request used to take 19 s (lazy loading), so everything is
  loaded at server start (about 40 s once).
- The first version of the generator was written by the project partner; we added the natural-language trigger,
  the replace question, the warm-up and the embedding fusion.

---

## 4. The LLM-first redesign

**Trigger:** feedback after the Part 2 presentation: use more LLM, the regexes are too hard-coded, and "the LLM
only generates keywords in R8".

**What we built:** `router.py`. For each message the LLM receives the message plus a short state summary (numbered
playlist, last song, open pick-list, last recommendations, pending question) and returns **one JSON action**
(add, remove, view, clear, help, ask, recommend, select, generate, confirm, feedback, preferences, forget, why,
chat, quit). Our existing, tested code carries the action out. Slash commands and button clicks skip the LLM.

**Why this shape:**
- The LLM can understand wording nobody wrote down ("nuke it all", "could you throw stnad by me on there") and
  resolve references from context ("the second one", "yes", "who made it").
- It cannot invent songs: it only chooses what to do, the database decides what exists.
- One call per message (median 0.5 s) leaves the 3-5 s and 5-10 call budgets intact.
- The old regex parsers remain as the fallback (no LLM, a call slower than 4 s, an unusable or invalid reply).

**How we got to 43/43** (`scripts/eval_router.py`, real LLM):
| Version | Result | Fix |
|---|---|---|
| first prompt, 42 messages | 39/42 | 3 misses were label or ambiguity problems |
| number the playlist | "who made the second one?" works | the model had passed "2" as a song name |
| ask for "Title by Artist" | 41/44 -> 43/43 | the model dropped the artist ("Yellow" instead of "Yellow by Coldplay") |

---

## 5. Tags: from a stop list to an LLM judgement

The catalog's tags are free Last.fm user text: 163,829 distinct tags, 99,386 of them used by one track.

| Attempt | Result |
|---|---|
| 13-word stop list ("favorites", "awesome" ...) | missed almost everything |
| hand-made opinion/year/name patterns + minimum frequency | removed noise but meant a hand-written list in the code |
| **LLM judges each tag once, offline** (`label_tags.py`) | 14,768 tags with >= 10 tracks judged in about 2 minutes; 5,331 kept as describing the music |

Why the LLM judgement: the lecturers did not want hard-coded knowledge, and a one-off offline judgement has no
run-time cost, is reproducible and can be read and corrected (`data/tag_labels.json`). The code contains no stop
list or filter patterns any more; a tag equal to the song's own title/artist/album is dropped.

Effect on quality (recommender, 1,337 turns): 0.2304 (old list) vs 0.2275 (judged tags): the same within noise,
with a vocabulary 30 times smaller (5,331 instead of 164,000). Visible effect: "Yellow" no longer shows
"loved, pretty, coldplay".

Where we deliberately did **not** use the judged tags: R8's search keeps the raw tags. BM25 over the judged tags
scored 0.0445 against 0.0613 for raw tags (fused: 0.0674 against 0.0727), because the raw tags also carry artist
names and other cues that help when a user names an artist.

---

## 5b. The web UI: two columns, playlist filter and catalog search

Added after Part 4 so the demo is easier to use.

| Before | Now | Why |
|---|---|---|
| the playlist under the chat in a half-width box | chat on the left, the playlist, taste chips and song search in a right column | the playlist is always in view; the whole width is used |
| an Add box that took text ("title by artist") | **Find songs**: search the whole catalog while typing, Add button per song | no need to know the exact title; shows which version (artist, album) you add |
| a long playlist could only be read by scrolling | a **filter box** inside the playlist | find a song in a long list |
| Clear removed everything at once | Clear asks first when there is more than one song | a misclick used to lose the whole playlist |

Decisions and reasons:
- **A plain `/search` HTTP route instead of the chat socket.** A search box answers on every keystroke; sending each
  one as a chat message would fill the conversation. The route sits on the same Flask app as the socket.
- **250 ms pause and cancelled requests.** The UI searches after the last key and ignores answers to outdated
  queries, so fast typing does not flood the server or show stale results.
- **Ranking:** every word must occur in the title or the artist, in any order; exact title first, then titles that
  start with the text, then popularity. "yellow coldplay" and "coldplay yellow" give the same list.
- **Typos only instead of nothing.** The fuzzy index offers "Stand By Me" for "stnad by me", but only when nothing
  matches exactly. The first version added close spellings after real matches and padded "yellow coldplay" with
  songs that do not contain "yellow"; seen in the browser, fixed, and covered by a test.
- **Add sends the track id** (`/add <id>`): a click can never pick a different song with the same title, and it goes
  through the same code as the chat.
- **Songs keep their real number under the filter,** so what the user sees matches "remove number 7".
- **Narrow screens / widget:** one column. Checked in a browser at desktop and 700 px width.
- Verified by 26 backend tests (ranking, typos, SQL safety, speed, the HTTP route) and 16 frontend tests (filter, delayed
  search, Add/Remove commands, empty and error states, taste chips), plus a manual run in the browser.
- Known issue found on the way: the frontend's Jest setup was already broken by a Babel version mismatch (see the note
  at the end of `implementation_log.md`).

## 6. Part 3: the interviews

- Two interviews (21 and 14 minutes), recorded on a phone, with verbal consent.
- Transcription options compared: Teams and Zoom need a subscription or university settings and do not separate
  speakers when two people share a laptop; online tools upload the audio. **Local Whisper** (`faster-whisper`,
  `scripts/transcribe.py`) is free, private and needs no ffmpeg.
- Lesson: automatic language detection was wrong for the second recording (it guessed Norwegian with 41%
  confidence and produced garbled text); forcing English gave a good transcript.
- Names were replaced by `[NAME]`; speaker labels were not required, so the transcripts stay as produced.
- Recordings and transcripts are kept out of git (`.gitignore`).

---

## 7. Part 4: feedback-aware refinement (all iterations)

Goal (the approved proposal): users refine recommendations in conversation ("I like the first two but the rest
are too heavy", "no more Coldplay", "make it more varied"); the system remembers, adapts and explains.

### Components
| Part | What it does | Why |
|---|---|---|
| Profile (`user_profile.py`) | liked/rejected tracks, artists, styles, wishes; the newest opinion wins | the memory of the session |
| Feedback understanding | the router's `feedback` action (positions, artists, styles, a wish, variety) | the LLM reads free wording; code checks and applies it |
| Implicit feedback | adding a suggested song = like, removing one = dislike | free signal from playlist actions |
| Rocchio | q' = q + 1.2 * mean(liked) - 0.3 * mean(disliked) over title/artist/album/tag terms; BM25 retrieves 1000 candidates | course material; changes **what is retrieved** |
| Neural reranker | trained two-tower network over Qwen3 metadata + collaborative-filtering embeddings | lecturer requirement; changes **the order** of the candidates |
| Style prototypes | a liked tag = the mean embedding of the tracks with that tag, scored in the trained space | tags enter only through embeddings |
| Constraint | explicitly rejected artists/styles are removed | a rejection should be final |
| MMR | diversity in the trained space; "more varied" -> lambda 0.3 over 200 candidates | user-controlled variety |
| Explanations | from the stored score components; before/after list difference; the LLM only words the opening sentence | grounded, not invented |
| Transparency | taste chips + Forget in the UI; "what do you know about me?", "forget my preferences", "why did you suggest X?" | the user can see and reset what the system learned |

### The reranker went through three designs
| Version | Idea | What happened |
|---|---|---|
| v1 (proposal v3) | rerank with BM25 + tag overlap + liked-artist match | worked (+14% with Rocchio; rejected tag share 0.86 -> 0.07) but the lecturer said BM25 is not reranking and tag reranking makes no sense |
| v2 (zero-shot neural) | cosine similarity to the playlist in the raw embedding spaces, request text as an anchor | **hurt**: 0.189 against 0.218 in a small test; plain cosine is noisy and the chat text is noisier |
| v3 (final) | **train** a two-tower network for "history -> next track" | held-out catalog-wide nDCG@20 0.213, recall@20 0.46 on its own |

Training (`train_reranker.py`): 95,725 turns from 90% of the sessions, sampled softmax (2,048 random negatives),
10 epochs, about 7 minutes on a laptop CPU; held-out recall@20 rose from 0.265 (epoch 1) to 0.463, nDCG@20 from
0.113 to 0.213. Every 10th session is never trained on and is used for the evaluation.

### Results (1,500 held-out turns; details in `part4_results.md`)
| nDCG@20, implicit feedback | all turns |
|---|---|
| Part 1 scoring (BM25 + priors) | 0.2251 |
| + Rocchio | 0.2580 |
| + neural reranker | 0.2372 |
| **+ Rocchio + neural reranker** | **0.2594** |
| raw cosines instead of the trained network | 0.2409 |
| nine mixes of the trained network with raw cosines | 0.249-0.260 (none better) |

Reading: Rocchio gives most of the gain; the neural reranker replaces the BM25 term at equal accuracy and carries
the feedback. The soft neural penalty cuts a rejected tag's share of the top 5 by 25-31% (artists 13-18%); the
constraint cuts 100%. Asking for a style raises its share from 0.13 to 0.25 and nDCG from 0.247 to 0.305.
MMR lambda 0.4 raises top-5 diversity from 0.265 to 0.302 for an nDCG cost of about 0.009.

### Choices inside Part 4 and the reason for each
- **Style prototypes instead of text embeddings of the tag word:** embedding the word hurt (nDCG 0.234 vs 0.277);
  the mean embedding of the tracks that carry the tag works. Weight 0.5 in the trained space was best (weight 2
  raised the share to 0.35 but nDCG fell to 0.243).
- **Rejections as a constraint:** with a Coldplay-only playlist the same-artist bonus (up to +3) beat a penalty of
  -2, so "no more Coldplay" still returned Coldplay. The soft penalty alone never reached 100%.
- **Liked artists and styles also expand the query:** otherwise they only reorder candidates that BM25 already
  found, and such songs may not be among them.
- **"More varied" = lambda 0.3 over 200 candidates:** at lambda 0.5 over 50 candidates the list did not change.
- **Wish text is embedded only when no tag matches:** zero-shot text matching is noisy.
- **`recommend` with a hint ("something more mellow") becomes a stated wish**, so it flows through the same code.

---

## 8. How we evaluated, and how we kept the evaluation honest

| Evaluation | Data | Rules we followed |
|---|---|---|
| Part 1 | train for tuning, test for reporting | weights on train; only K and a few lam values compared on test (disclosed) |
| R6 recommender | 1,337 train turns, two halves | weights from half 1, checked on half 2 |
| Part 4 ranking | 1,500 turns from sessions the network never saw | weights tuned on 400 turns of the first half; both halves reported |
| Router | 43 labelled messages | regression test for prompt changes |
| Feedback parsing | three sets of 30 sentences | see below |

**Feedback parsing, step by step** (`scripts/eval_feedback.py`): the first 30 sentences scored 100% after tuning
the prompt on them. A second set of 30 (acceptable readings written down before running) scored **80%**, below the
promised 85% - the first sign that we were optimistic. We made one principled prompt fix (opinions about listed
songs are feedback, not a select), fixed a too-strict checker ("heavier" did not match "heavy"), added two
equivalence rules (a `select` counts as "liked these" because adding a suggested song registers a like; a
`recommend` hint counts as a wish), then wrote a third set and ran it **once with nothing changed afterwards**.

| Set | Role | Result |
|---|---|---|
| tuned | the prompt was tuned on it | 28/30 (93%) |
| second | seen once, one principled fix | 27/30 (90%) |
| fresh | run once, no changes after | **28/30 (93%)** |

Disclosures: the equivalence rules were added after seeing results (they apply to every sentence); all sentences
were written by the author of the prompt, so sentences from someone else would be a stronger test; a test checks
that the three sets do not overlap.

Scenario evaluations in Part 4 (rejecting a tag or artist, asking for a style) are **constructed**: they pick
attributes the target does or does not have, so nDCG changes in them by construction. Only the shares are the
point. No evaluation used real users.

---

## 9. Bugs and corrections that taught us something
| What went wrong | What we learned |
|---|---|
| `self._pending_multi = self._pending_then = self._add_chosen` (a chained assignment instead of `==`) overwrote the pending callback; two tests failed | type errors in the editor and failing tests pointed at the same line |
| a recommender `SELECT` lost the `track_name` column in a merge, so every column index shifted | after merging, run the whole suite |
| shell quoting turned `\b` into invisible control characters in a regex | scan source files for control characters after generated edits |
| the router dropped the artist from song names | state the output format in the prompt ("Title by Artist") |
| the router read "like these" as "more of the same" and "a few more" as "more varied" | only set `variety` when it is explicitly asked for; re-run the router tests after each prompt edit |
| the confirmation text had `<b/>` instead of `</b>` and "16 song" | a test now checks the HTML and the plural |
| first request after start-up took 19 s | load heavy objects at server start |
| an evaluation scenario placed the style word in a long chat query and nothing changed | the chat uses the profile as the query, so the evaluation must too |
| language auto-detection guessed Norwegian for an English interview | force the language when it is known |
| `profile.py` would shadow Python's standard `profile` module | pick file names that do not clash with the standard library |

---

## 10. Everything we tried that did not make it
| Idea | Where | Result | Why dropped |
|---|---|---|---|
| current message as the only query | Part 1 | 0.053-0.080 | the history matters far more |
| `tag_list` in the Part 1 BM25 | Part 1 | no gain | metadata already in the query |
| current-message BM25 in the score | Part 1 | worse | w2 = 0 |
| protecting top ranks from the usage penalty | Part 1 | worse | keeps hub tracks |
| lam = 2.0 | Part 1 | nDCG drops | over-penalises |
| dense retrieval, fusion, learned reranker | Part 1 | not tried | target already beaten; returned in Parts 2 and 4 |
| BM25 over the playlist for R6 | Part 2 | no clear gain | other features cover it |
| embedding centroids for R6 | Part 2 | 0.146 vs 0.219 | worse than the metadata features |
| tag-only or names-only BM25 for R8 | Part 2 | 0.0315 / 0.0414 vs 0.0613 | real requests name artists and songs |
| judged tags in the R8 search | Part 2 | 0.0445 vs 0.0613 | loses artist-name cues |
| 2:1 weight for embeddings over BM25 in R8 | Part 2 | more literal results | 1:1 kept |
| prompt that forbids all occasion words | Part 2 | dropped "love" from "sad love songs" | final wording keeps themes |
| Teams/Zoom transcripts | Part 3 | subscription / no speaker split | local Whisper |
| BM25 + tag-overlap reranker | Part 4 v3 | worked, but not allowed | lecturer feedback |
| zero-shot cosine reranker | Part 4 | 0.19 vs 0.22 | the network must be trained |
| request text as a reranking anchor | Part 4 | hurt | noisy chat text |
| text embedding of the style word | Part 4 | 0.234 vs 0.277 | style prototypes |
| mixing raw cosines into the trained network | Part 4 | 0.249-0.260 | no better than the network alone |
| soft penalty as the only rejection | Part 4 | 18-31% cut | constraint |
| an LLM-written reason for every suggested song | Part 4 | not built | reasons computed from the score components are faster and cannot be invented; the LLM only words one opening sentence |

---

## 11. Lessons
1. Measure a data property before designing features (the 60.7% artist continuity drove Parts 1 and 2).
2. Remove what can never be the answer (played tracks) before anything clever.
3. An LLM is most useful for *understanding*, and safest when code does the *doing* against a database.
4. "Neural" has to be trained for the task: a plain cosine in a pretrained space was worse than the baseline.
5. Fix the evaluation before trusting a number: scenarios must mimic how the system is used (the chat query is the
   profile, not the conversation text).
6. Write the acceptable answers *before* running a test set, and run a fresh set once.
7. Prompts regress silently: keep a labelled test set and re-run it after every edit.
8. Keep every experiment script and its output so a number can be reproduced.

---

## 12. Limits and open items
- Rocchio gives most of the accuracy gain; the neural reranker mainly replaces BM25 scoring at equal accuracy.
- The soft neural penalty alone cuts a rejected attribute by 13-31%; only the constraint makes it complete.
- Scenario evaluations are constructed; nothing was tested with real users.
- With the LLM on, every message costs a model call; without it the regex fallback understands fewer wordings.
- Feedback is session-level: nothing is stored across chats.
- The tag labels are one LLM's judgement; a few borderline tags were kept.
- Song titles go into the router prompt: actions are validated and only the database supplies songs, but a hostile
  title could influence the wording of a reply.
- Typical misses of the feedback parser: "just more of what you gave me" (variety "less"), "give me upbeat summer
  songs" and "I'm in the mood for jazz" (read as generate a playlist), a 4 s timeout.
- No type checker (Pyright) was run; Pylance may show warnings.
- The first run on a new machine needs the embedding caches (about 0.8 GB) and `data/reranker.pt` (committed).

---

## 13. Who did what, and how to reproduce

**Who did what (fill in before submitting).** From the working history: the Part 2 core (R1-R7 code) was written by
the repository owner with step-by-step guidance; the project partner wrote the first version of the R8 playlist
generator and its script; the router, the tag judgement, Part 4, the evaluation scripts, the tests and these
documents were written with the AI coding assistant Claude Code at the owner's request. State this as the course
requires.

**Reproduce**
```
python -m scripts.import_tracks                  # build data/tracks.db
python -m scripts.label_tags                     # (optional) re-judge the tags -> data/tag_labels.json
python -m scripts.train_reranker                 # (optional) retrain -> data/reranker.pt, about 7 minutes
python experiments/10_refine_eval.py --turns 1500 --scenarios 400 --tune     # Part 4 numbers, about 1 hour
python -m scripts.eval_router                    # router accuracy (needs MY_OLLAMA_API_KEY)
python -m scripts.eval_feedback --set fresh      # feedback parsing; also --set tuned / heldout
python -m pytest -q                              # 587 tests
cd frontend && CI=true npx react-scripts test --watchAll=false   # 16 frontend tests
python musiccrs/musiccrs.py                      # backend; frontend: cd frontend && npm start
```

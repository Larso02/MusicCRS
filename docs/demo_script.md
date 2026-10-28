# MusicCRS demo script (about 6 minutes)

The commands below were run on the real system with the LLM on (LLM-first, rules as the fallback).
Wording can vary - the LLM understands paraphrases - but these exact messages worked.
Recommendations and generated playlists differ a little between runs.

## Before the demo (do this 10 minutes early)

1. Backend, from the repo root (takes 30-40 s to warm up, wait for it):
   ```
   conda activate musiccrs
   python musiccrs/musiccrs.py
   ```
2. Frontend, in a second terminal:
   ```
   cd frontend
   npm start
   ```
   Open http://localhost:3000 and sign in (Anonymous works).
3. `.env` must contain `MY_OLLAMA_API_KEY` - the LLM understands every message and writes R8's
   search. `data/tag_labels.json` must exist (it is in the repo). The first full start has to have
   downloaded the embedding model and the track embeddings.
4. Send one test message and one `make me a playlist for studying` (answer `yes`), then `clear it all`.
   The first requests after a fresh start are the slowest; do it before the examiner arrives.
5. Start the demo with an **empty playlist**.

## Part 1: playlist control (R1, R2, R3, R4), 1.5 min

| Say or do | What happens | Point to make |
|---|---|---|
| `hey there` | a short friendly reply | **LLM front door**: also handles chat, not just commands |
| `what can you do for me?` | list of all features | **R3**: help in natural language |
| `add Clocks by Coldplay` | added; the playlist panel updates | **R2** add by title and artist |
| `could you throw stnad by me on there` | asks which *Stand By Me* (3 versions), pick 1 | free wording and a typo, understood; **R4** ranked list with buttons |
| `add the nirvana song with the teen spirit in the title` | adds *Smells Like Teen Spirit - Nirvana* | loose reference, no exact title needed |
| click a button in a pick-list | the song is added | buttons and typing behave the same |
| type a song in the **Add box** under the panel | added | **R1**: same actions directly from the UI |
| click **Remove** next to a song | removed | **R1** |
| `get rid of the first one` | removes the first song | references to the playlist ("the first one") |
| `what do I have so far?` | numbered list | view in natural language |

## Part 2: questions (R5), 1 min

With songs on the playlist (for example Clocks, Smells Like Teen Spirit, Heart-Shaped Box):

- `who made the second one?` -> Smells Like Teen Spirit is by Nirvana (resolved from the playlist)
- `how long is the last one?` -> 4:41 (Heart-Shaped Box)
- `what genre is Yellow by Coldplay?` -> top descriptive tags (rock, alternative, pop ...)
- `what style is Nirvana?` -> grunge, rock, alternative ...
- `how many songs does Coldplay have?`
- `tell me about Nirvana`

Point to make: the LLM only decides *which question* was asked; the answer is read from the database,
so nothing is invented. 11 kinds of questions were implemented (4 required). Tags are filtered by an
LLM judgement made once offline (no stop list in the code).

## Part 3: recommendations and selection (R6, R7), 1.5 min

| Say | What happens | Point to make |
|---|---|---|
| `give me some songs like these` | 5 songs, each with a reason ("same artist", "similar style: ...") | **R6**: related to the playlist, not random or just popular |
| `add the first and the last` | adds those two | **R7** formulation 1: positions |
| `give me some songs like these` | 5 new songs | no repeats |
| `everything except the last one` | adds 4 | **R7** formulation 2: exclusion |
| `give me some songs like these`, then `the ones by Nirvana` | adds all Nirvana songs in the list | **R7** formulation 3: by artist |
| `give me some songs like these`, then `nah skip those` | nothing added | "no songs at all" |
| `give me some songs like these`, then `add them all` | all 5 added | "all songs" |

Say it out loud: the five songs are scored on artist, album, popularity, tag similarity and release
year (Part 1 features plus tags): nDCG@20 0.2317 against 0.2205 without tags and year.

## Part 4: generated playlists (R8), 1 min

| Say | What happens | Point to make |
|---|---|---|
| `make me a playlist for the gym` | with a non-empty playlist: asks "replaces your current playlist (N songs). Say yes or no" | protects the user's playlist |
| `no` | "Okay, I'll keep your current playlist." | |
| `make me a playlist for the gym`, then `yes` | about 13 songs, about 62 minutes | **R8**: the system chooses the length |
| `I need something for a long road trip`, then `yeah go for it` | about 31 songs, about 120 minutes | the LLM reads a free-form answer; a longer activity gives a longer playlist |
| `show me what I have` | shows the generated list | |
| `clear it all`, then `ok I'm done, bye` | empty playlist, then the chat ends | |

Point to make: one LLM call turns the request into keywords and a duration; candidates come from
BM25 (our Part 1 retriever) fused with embedding retrieval; ids are checked against the database.
BM25 alone scored 0.0655, with embeddings 0.0829.

## If something goes wrong

- **A reply takes more than a few seconds:** the model was slow; the system falls back to the rule
  parsers after 4 s. The first R8 request after start-up is also slower - run it before the demo.
- **"Playlist generation requires an LLM" / replies feel stiff:** the key is missing or the model
  server is down. The rule-based parsers still handle the standard wordings (`add ...`, `show playlist`,
  `recommend something`); explain the design and show R8 from the code (`musiccrs/playlist_generator.py`).
- **"I'm sorry, I don't understand":** rephrase, or use a standard wording from the tables.
- **Buttons do not appear:** type the number instead; it goes through the same code.

## Likely examiner questions

- *Where do you use the LLM?* In front of everything: it reads each message (with the playlist, last
  song, open list and pending question) and returns one action; and in R8 to write the search
  keywords and the playlist length. It also judged the catalog's tags once, offline.
- *Does the LLM recommend songs?* No. Songs come from our database and retrievers; every id is
  checked against the database before it reaches the playlist.
- *What if the LLM is wrong or slow?* The action is validated (known action, required fields, numbers
  inside the open list). If not, or after 4 s, the rule-based parsers answer.
- *Why not use the LLM for the answers too?* Facts (year, length, popularity, tags) must be exact;
  they are read from the database. The LLM decides *what* was asked.
- *How is "loose reference" handled?* The LLM extracts title and artist, then typo-tolerant search
  with two thresholds: 0.9 accepts, 0.6 asks "Did you mean".
- *How do you rank disambiguation lists?* popularity/100 plus bonuses for artist and album already on
  the playlist.
- *How are you tested?* 478 automated tests over the real catalog, a scripted fake LLM for the
  router, a 43-message test against the real model (43/43), and a scripted conversation of all
  requirements with timings.
- *What is hard-coded?* Only the rule-based fallback, the action list the LLM can choose from, and a
  small mapping from the LLM's question kinds to the answer code. The tag filter is an LLM judgement
  stored in a data file, not code.

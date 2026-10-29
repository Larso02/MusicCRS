# Part 4 proposal draft (v4, after the third lecturer feedback)

## Text for Canvas (under 300 words)

Feedback-aware playlist refinement

Idea. Users refine recommendations conversationally ("I like the first two, but too heavy; make it more varied"). One LLM call turns each feedback sentence into structured positives and negatives (tracks, artists, styles), kept in a session profile. Three stages build on our Part 1 BM25 retriever:
1. Rocchio relevance feedback (Rocchio, 1971) modifies the query: q' = αq + β·mean(liked) − γ·mean(disliked). BM25 retrieves the top 1000 candidates for q'.
2. A neural reranker rescores them: a two-tower network trained on TalkPlay sessions over the challenge's Qwen3 metadata and collaborative-filtering embeddings. Liked styles enter as prototypes (mean embedding of tracks with the tag), wishes as text embeddings, rejected songs and styles as penalties. Explicitly rejected artists and styles are removed as a constraint.
3. Maximal Marginal Relevance (Carbonell & Goldstein, 1998) diversifies the list; "more varied" sets its strength. Adding or removing a recommended track is implicit feedback.

Evaluation. Offline on held-out TalkPlay sessions: nDCG@20 for Part 1 scoring, Rocchio, neural reranker and both (ablation); scenarios for rejected attributes, requested styles, diversity, parsing accuracy.

General condition: responses within 3-5 s; ranking needs no LLM; at most 3 LLM calls per turn.

| Planned feature | Points |
|---|---|
| Parse feedback via LLM; at least 85% correct on 30 sentences | 2 |
| Session preference profile updated over several turns | 2 |
| Rocchio query modification and trained neural reranking; rejected-attribute rate in top-5 cut by at least 30% while keeping relevance | 4 |
| MMR diversity with user-controlled strength | 2 |
| Implicit feedback from playlist add/remove | 1 |
| Grounded explanations from score components plus before/after diff | 2 |
| Evaluation vs. Part 1 baseline with ablation | 2 |
| Total | 15 |

---

## Background (not for Canvas)

### Lecturer feedback and how v4 answers it
| Feedback | Change |
|---|---|
| (v2) Rocchio goal quite ambitious | Threshold softened from 50% to 30% rejected-attribute reduction. |
| (v2) Parsing looks like 2p since evaluation is scored separately | Parsing 3p -> 2p. |
| (v2) Add one more small 1p task | Implicit feedback from playlist actions. |
| (v2) Latency should be a general condition | Stated as a general condition, not a table row. |
| (v2) Explanations: add value beyond asking the LLM | Explanations are computed from score components plus a before/after diff. |
| (v3) "Rocchio is not reranking" | Rocchio is described as query modification before retrieval. |
| (v3) Rerank with BM25 and tags, and Rocchio | (replaced in v4, see below) |
| (v4) "BM25 is not used for reranking; tag reranking does not make sense; reranking needs a neural method, tags may be added" | The reranker is now a **trained neural network** (two-tower, Qwen3 metadata + collaborative-filtering embeddings). BM25 only retrieves candidates (with the Rocchio query). Tags are no longer counted: a liked tag selects the tracks whose mean embedding forms a style prototype, which the network space scores by cosine similarity. Explicit rejections are a constraint, not a score. |

### How the stages fit together
- Rocchio changes **what is retrieved** (the BM25 query vector q'); the neural reranker changes **the order of what was retrieved** (the top K = 1000 candidates). They are separate components with separate tests.
- The neural reranker (`musiccrs/neural.py`, `musiccrs/reranker_model.py`) is a two-tower network trained by `scripts/train_reranker.py` on 90% of the TalkPlay training sessions (history of played tracks -> next track, sampled softmax). Inputs: the Qwen3 metadata embedding (1024-d) and the BPR collaborative-filtering embedding (128-d) that the challenge data provides. The other 10% of the sessions are held out for the evaluation.
- Profile effects in the reranker: liked and played songs form the history; liked tags become style prototypes (mean embedding of the tracks with the tag, scored in the trained space); wishes that match no tag are embedded as text with the same Qwen3 model; rejected songs and styles are penalties; explicitly rejected artists and styles are removed from the candidates (constraint).
- Priors from Part 1 that are neither BM25 nor tags (same artist/album as the playlist, popularity) can be added; the ablation shows whether they help.
- MMR uses cosine similarity in the trained space.

### Why this proposal
- Relevance feedback (Rocchio) is course material, same family as the RM3 pseudo-relevance feedback in assignment A9; MMR is a standard citable diversity method.
- Part 1 evidence (`docs/part1_experiment_log.md`): implicit positive feedback (previously played tracks, same artist/album) improved ranking by 20-44% and the target's artist already appeared in 60.7% of turns. Explicit positive and negative feedback is the natural generalisation.
- Part 2 evidence (`musiccrs/recommender.py`): on 1,337 train turns, artist/album/popularity scored nDCG@20 0.2205 and adding tag similarity and release year gave 0.2317, so tags are a useful reranking signal. BM25 over the playlist metadata gave no clear gain there (there was no text query); the feedback sentences supply the text query that BM25 needs.

### Evidence from the feasibility run (`experiments/09_retrieval_feasibility.py`, 1,500 train turns, nDCG@20)
- Text-only request: BM25 0.0655; metadata embedding 0.0645; BM25 + metadata embedding (RRF) 0.0829 (+27%); attributes/lyrics embeddings weak and hurt the fusion.
- Playlist-based next track: artist/album/popularity heuristic 0.2190; embedding centroids alone: metadata 0.1457, CF 0.1150, attributes 0.0676, audio 0.0348; crude heuristic + embedding combination gave no gain.
- Latency on a 2015 laptop CPU: 0.4-0.7 s per query embedding (float32; default bfloat16 ~100x slower), 84 ms for a 47k x 1024 scan.

### Notes for implementation
- No "genre" field exists: use `tag_list`.
- Rocchio on sparse vectors: represent each track as TF-IDF over its title/artist/album/tag terms (the same terms the BM25 index uses); q' keeps only the top terms by weight and clips negative weights to zero, since BM25 cannot score negative query weights. The negative part is therefore applied in the neural reranker (penalty) and as a constraint.
- Implicit feedback needs the playlist actions to carry the recommended track ids (the Part 2 playlist already stores `track_id`).
- Explanations: log per-track score components (trained network, style prototype, wish text, artist/album prior) so reasons are real, not generated.
- Start collecting the 30 labelled feedback sentences and 5-10 scenarios now.
- Priority if time is short: keep Rocchio + neural reranker + profile + evaluation; drop MMR strength parsing and the list diff first.

### References for the Canvas text
- Rocchio, J. J. (1971). Relevance feedback in information retrieval. In *The SMART Retrieval System*, pp. 313-323. Prentice-Hall.
- Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *SIGIR '98*, pp. 335-336.

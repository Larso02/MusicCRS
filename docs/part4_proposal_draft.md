# Part 4 proposal draft (v3, after second lecturer feedback)

## Text for Canvas (under 300 words)

Feedback-aware playlist refinement

Idea. Users refine recommendations conversationally ("I like the first two, but too heavy; make it more varied"). One LLM call turns each feedback sentence into structured positives and negatives (tracks, artists, tags), kept in a session profile. Two stages then build on our Part 1 BM25 retriever:
1. Rocchio relevance feedback (Rocchio, 1971) modifies the query: q' = αq + β·mean(liked) − γ·mean(disliked), over title, artist, album and tag terms. BM25 retrieves candidates for q'.
2. A feedback-aware reranker rescores the top-1000 candidates using BM25 for q', tag overlap with liked tags, artist/album match, and a penalty for rejected artists and tags; weights tuned on train sessions.
Maximal Marginal Relevance (Carbonell & Goldstein, 1998) then diversifies the list; "more varied" sets its strength. Adding or removing a recommended track gives implicit feedback.

Evaluation. Offline on TalkPlay train sessions (previously played tracks as implicit feedback): nDCG@20 for Part 1 alone, Rocchio only, reranking only and both. Scripted scenarios measure rejected attributes in the top 5, intra-list diversity and parsing accuracy.

General condition: responses within 3-5 s; ranking uses no LLM; at most 3 LLM calls per turn.

| Planned feature | Points |
|---|---|
| Parse positive/negative feedback via LLM; at least 85% correct on 30 sentences | 2 |
| Session preference profile updated over several turns | 2 |
| Rocchio query modification and BM25-and-tag reranking; rejected-attribute rate in top-5 cut by at least 30% while keeping relevance | 4 |
| MMR diversity with user-controlled strength | 2 |
| Implicit feedback from playlist add/remove | 1 |
| Grounded explanations computed from score components plus before/after list diff | 2 |
| Evaluation vs. Part 1 baseline with ablation (nDCG@20, rejected-attribute rate, diversity) | 2 |
| Total | 15 |

---

## Background (not for Canvas)

### Lecturer feedback and how v3 answers it
| Feedback | Change |
|---|---|
| (v2) Rocchio goal quite ambitious | Threshold softened from 50% to 30% rejected-attribute reduction. |
| (v2) Parsing looks like 2p since evaluation is scored separately | Parsing 3p -> 2p. |
| (v2) Add one more small 1p task | Implicit feedback from playlist actions. |
| (v2) Latency should be a general condition | Stated as a general condition, not a table row. |
| (v2) Explanations: add value beyond asking the LLM | Explanations are computed from score components plus a before/after diff. |
| (v3) "Rocchio is not reranking" | Rocchio is now described as what it is: query modification before retrieval. It is no longer called re-ranking. |
| (v3) Implement both reranking with BM25 and tags, and Rocchio, for 4p | The 4p row now contains both: Rocchio query modification, plus a separate reranking stage over the retrieved candidates using BM25 and tag scores. The ablation in the evaluation row shows what each part contributes. |

### How the two stages fit together
- Rocchio changes **what is retrieved** (the query vector q'); the reranker changes **the order of what was retrieved** (scoring the top-K, K = 1000 as in Part 1). They are separate components with separate tests.
- Both use the same profile (liked and rejected tracks, artists, tags), so one feedback sentence updates both.
- Tags come from `tag_list`. The tags are noisy user tags, so only descriptive ones should be used (drop opinion tags such as "favorites", single years and the artist's own name, and tags used by fewer than 5 tracks). A tested version of this filter exists but is not merged into the Part 2 code yet.
- Optional extension, only if time remains: add the TalkPlay metadata embedding as a fourth reranking signal. It is not promised in the table.

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
- Rocchio on sparse vectors: represent each track as TF-IDF over its title/artist/album/tag terms (the same terms the BM25 index uses); q' keeps only the top terms by weight and clips negative weights to zero, since BM25 cannot score negative query weights. The negative part is therefore applied mainly in the reranker (rejected penalty).
- Implicit feedback needs the playlist actions to carry the recommended track ids (the Part 2 playlist already stores `track_id`).
- Explanations: log per-track score components (BM25 term, tag match, artist/album match, penalty, MMR term) so reasons are real, not generated.
- Start collecting the 30 labelled feedback sentences and 5-10 scenarios now.
- Priority if time is short: keep Rocchio + reranker + profile + evaluation; drop MMR strength parsing and the list diff first.

### References for the Canvas text
- Rocchio, J. J. (1971). Relevance feedback in information retrieval. In *The SMART Retrieval System*, pp. 313-323. Prentice-Hall.
- Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *SIGIR '98*, pp. 335-336.

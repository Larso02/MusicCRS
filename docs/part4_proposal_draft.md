# Part 4 proposal draft (v2, after lecturer feedback)


**Feedback-aware playlist refinement**

**Idea.** Users refine recommendations conversationally ("I like the first two, but too heavy; more alternative rock", "make it more varied"). One LLM call turns each feedback sentence into structured positives and negatives (tracks, artists, tags), accumulated in a session profile. Candidates from our Part 1 retriever are re-ranked with **Rocchio relevance feedback** (Rocchio, 1971): q' = αq + β·mean(liked) − γ·mean(disliked), over tags/artists plus TalkPlay metadata embeddings. **Maximal Marginal Relevance** (Carbonell & Goldstein, SIGIR 1998) trades relevance against similarity to selected tracks; "more varied" sets its strength. Playlist actions also give implicit feedback: adding a recommended track counts as positive, removing it as negative.

**Evaluation.** Offline on TalkPlay train sessions, treating previously played tracks as implicit feedback: nDCG@20 against our Part 1 retriever. Scripted scenarios: share of top-5 results with a rejected tag/artist before vs. after feedback; intra-list diversity before vs. after MMR; parsing accuracy on 30 labelled sentences.

**General condition.** Responses within 3-5 s: ranking needs no LLM, at most 3 LLM calls per turn.

| Planned feature | Points |
|---|---|
| Parse positive/negative feedback (tracks, artists, tags) via LLM; at least 85% correct on 30 labelled sentences | 2 |
| Session preference profile, updated over several turns | 2 |
| Rocchio re-ranking; rejected-attribute rate in top-5 cut by at least 30% while keeping relevance | 4 |
| MMR diversity with user-controlled strength | 2 |
| Implicit feedback from playlist actions (add/remove of recommended tracks) updating the profile | 1 |
| Grounded explanations: per-track reason computed from score components (liked artist/tag matched, penalised attribute) plus a before/after list diff; the LLM only phrases them | 2 |
| Quantitative evaluation vs. Part 1 baseline (nDCG@20, rejected-attribute rate, intra-list diversity) | 2 |
| **Total** | **15** |

---

## Background (not for Canvas)

### Lecturer feedback and how v2 answers it
| Feedback | Change |
|---|---|
| Rocchio goal quite ambitious | Threshold softened from 50% to 30% rejected-attribute reduction; representation kept simple (tags/artists + metadata embedding). |
| Task 1 (parsing) looks like 2p since evaluation is scored separately | Parsing 3p -> 2p. |
| Add one more small 1p task | Implicit feedback from playlist actions (add = positive, remove = negative). |
| Latency should be a general condition | Removed from the table; stated as a general condition. |
| Explanations: keep, but add value beyond asking the LLM | Explanations are computed from score components (matched liked artist/tag, penalised attribute) plus a before/after diff; the LLM only phrases them. 2p. |

### Why this proposal
- Relevance feedback (Rocchio) is course material, same family as the RM3 pseudo-relevance feedback in assignment A9; MMR is a standard citable diversity method.
- Part 1 evidence (`docs/part1_experiment_log.md`): implicit positive feedback (previously played tracks, same artist/album) improved ranking by 20-44% and the target's artist already appeared in 60.7% of turns. Explicit positive and negative feedback is the natural generalisation.

### Evidence from the feasibility run (`experiments/09_retrieval_feasibility.py`, 1,500 train turns, nDCG@20)
- Text-only request: BM25 0.0655; metadata embedding 0.0645; BM25 + metadata embedding (RRF) 0.0829 (+27%); attributes/lyrics embeddings weak and hurt the fusion.
- Playlist-based next track: artist/album/popularity heuristic 0.2190; embedding centroids alone: metadata 0.1457, CF 0.1150, attributes 0.0676, audio 0.0348; crude heuristic + embedding combination gave no gain.
- Latency on a 2015 laptop CPU: 0.4-0.7 s per query embedding (float32; default bfloat16 ~100x slower), 84 ms for a 47k x 1024 scan.
- Consequence: profile on artist/album/tags plus the **metadata** embedding only.

### Notes for implementation
- No "genre" field exists: use `tag_list` and embeddings.
- Implicit feedback needs the playlist actions to carry the recommended track ids (ties in with the Part 2 playlist record: `track_id`, name, artist).
- Explanations: log per-track score components (artist match, tag match, penalty, MMR term) so reasons are real, not generated.
- Start collecting the 30 labelled feedback sentences and 5-10 scenarios during Part 2.
- Priority if time is short: keep Rocchio + profile + evaluation; drop MMR strength parsing and the list diff first.

### References for the Canvas text
- Rocchio, J. J. (1971). Relevance feedback in information retrieval. In *The SMART Retrieval System*, pp. 313-323. Prentice-Hall.
- Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *SIGIR '98*, pp. 335-336.

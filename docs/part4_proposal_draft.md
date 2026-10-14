# Part 4 proposal draft



**Feedback-aware playlist refinement**

**Idea.** Users refine recommendations conversationally ("I like the first two, but too heavy; more alternative rock", "make it more varied"). One LLM call converts each feedback sentence into structured positives and negatives (tracks, artists, tags), which a session profile accumulates across turns. Candidates from our Part 1 retriever are re-ranked with **Rocchio relevance feedback** (Rocchio, 1971): q' = αq + β·mean(liked) − γ·mean(disliked), over a combined representation: BM25 fields including tags, plus the TalkPlay metadata embeddings. A **Maximal Marginal Relevance** step (Carbonell & Goldstein, SIGIR 1998) trades relevance against similarity to already selected tracks; requests such as "more varied" set its strength. Ranking uses no LLM, so responses stay within 3-5 s, and the system briefly explains how the feedback changed the list.

**Evaluation.** Offline on TalkPlay train sessions, with previously played tracks as implicit feedback: nDCG@20 against our Part 1 retriever. Scripted scenarios: share of top-5 results containing a rejected tag/artist before vs. after feedback; intra-list diversity before vs. after MMR; parsing accuracy on 30 labelled sentences.

| Planned feature | Points |
|---|---|
| Parse positive/negative feedback (tracks, artists, tags) via LLM; at least 85% correct on 30 labelled sentences | 3 |
| Session preference profile, updated over several turns | 2 |
| Rocchio re-ranking (BM25/tags + embeddings); rejected-attribute rate in top-5 cut by at least 50% while keeping relevance | 4 |
| MMR diversity with user-controlled strength | 2 |
| Quantitative evaluation vs. Part 1 baseline (nDCG@20, rejected-attribute rate, intra-list diversity) | 2 |
| Latency up to 5 s, at most 3 LLM calls per turn; short explanation of the feedback effect | 2 |
| **Total** | **15** |


---

## Background

### Why this combined proposal
- Merges the two proposals from the project partner (relevance feedback + diversity-aware re-ranking) into one feature, because both are re-ranking stages over the same retriever, and two people cannot build two full features alongside Part 2 and Part 3.
- Rocchio relevance feedback is core course material (same family as the RM3 pseudo-relevance feedback from assignment A9), and MMR is a standard citable diversity method.
- Motivated by our Part 1 findings: implicit positive feedback (previously played tracks, same artist/album) improved ranking by 20-44% and the target's artist already appeared in 60.7% of turns (see `docs/part1_experiment_log.md`). Explicit natural-language feedback, positive and negative, is the natural generalisation.

### Evidence from the feasibility run (`experiments/09_retrieval_feasibility.py`, 1,500 train turns, nDCG@20)
- Text-only request: BM25 0.0655; metadata embedding alone 0.0645; BM25 + metadata embedding (RRF) 0.0829 (+27%); attributes and lyrics embeddings weak (0.0246, 0.0131) and hurt the fusion.
- Playlist-based next track: artist/album/popularity heuristic 0.2190; embedding centroids alone: metadata 0.1457, CF 0.1150, attributes 0.0676, audio 0.0348; crude heuristic + embedding combination gave no gain (0.2199).
- Latency on a 2015 laptop CPU: 0.4-0.7 s to embed one query (float32; the default bfloat16 was ~100x slower), 84 ms for a 47k x 1024 similarity scan.
- Consequence: build the profile on symbolic features (artist, album, tags) plus the **metadata** embedding only; do not rely on attributes/lyrics/audio embeddings.

### Risks and how the rubric handles them
- Thresholds (85% parsing accuracy, 50% reduction in rejected-attribute rate) are chosen to be achievable: a 31B model parses feedback well, and a hard-ish penalty on rejected tags/artists reduces their share directly. Revisit them with the lecturer if you want different numbers.
- There is no "genre" field in the catalog: use `tag_list` (Last.fm-style tags such as "relaxing", "piano") and embeddings. Do not write "genre" in the final text.
- Scope: Part 2 (due Oct 21/28) and Part 3 (Oct 28) come first. If time gets tight, drop the explanation feature and the "more varied" parsing first; keep the Rocchio core and the evaluation.
- The "rejected-attribute rate" and "intra-list diversity" evaluations need hand-written scenarios: start collecting 30 feedback sentences and 5-10 scenarios during Part 2 so they exist when implementation starts.


### References for the Canvas text
- Rocchio, J. J. (1971). Relevance feedback in information retrieval. In *The SMART Retrieval System*, pp. 313-323. Prentice-Hall.
- Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *SIGIR '98*, pp. 335-336.

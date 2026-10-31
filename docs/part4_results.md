# Part 4: feedback-aware refinement - what was built and what it achieves

All numbers come from `experiments/10_refine_eval.py` (1,500 train turns from **held-out sessions** the
neural reranker never saw; no test data) and from the scripts named in each section.

## The pipeline

1. **Profile** (`musiccrs/user_profile.py`): liked and rejected tracks, artists, styles (tags) and wishes,
   updated by every feedback sentence. The newest opinion wins. Adding a suggested song counts as a like,
   removing one as a dislike (implicit feedback).
2. **Understanding the feedback** (`musiccrs/router.py`): the LLM turns "I like the first two but the rest
   are too heavy" into liked/rejected positions, artists, styles, a wish and a variety request.
3. **Rocchio query modification** (`TermIndex.rocchio`): q' = q + 1.2 * mean(liked) - 0.3 * mean(disliked),
   term vectors over title, artist, album and tag words; negative weights are clipped, the top 40 terms kept.
   BM25 retrieves the top 1,000 candidates for q'. BM25 is used for retrieval only.
4. **Neural reranker** (`musiccrs/neural.py`, `musiccrs/reranker_model.py`): a two-tower network, trained by
   `scripts/train_reranker.py` on 90% of the TalkPlay training sessions (history of tracks -> next track,
   sampled softmax, 10 epochs, about 7 minutes on a laptop CPU). Inputs are the challenge's Qwen3 metadata
   embedding (1024-d) and BPR collaborative-filtering embedding (128-d). The score is the cosine similarity
   between the user tower (playlist and liked songs) and the track tower, z-scored over the candidates.
   - liked styles: the mean embedding of the tracks with the tag (a style prototype), scored in the same space;
   - wishes that match no tag: embedded as text with the same Qwen3 model;
   - rejected songs and styles: penalties for above-average similarity;
   - small priors from Part 1 that are neither BM25 nor tags: same artist or album as the playlist, popularity;
   - explicitly rejected artists and styles are removed as a constraint.
5. **MMR** with similarity in the trained space; "make it more varied" sets lambda to 0.3.
6. **Explanations**: computed from the score components, with a before/after list difference; the LLM only
   words the opening sentence.

No tag is counted anywhere in the reranker, and BM25 plays no part in it.

## Results

### A) Implicit feedback (nDCG@20, 1,500 held-out turns)
| Variant | first half | second half | all |
|---|---|---|---|
| Part 1 scoring (BM25 + priors) | 0.2280 | 0.2222 | 0.2251 |
| + Rocchio | 0.2566 | 0.2593 | 0.2580 |
| + neural reranker | 0.2394 | 0.2351 | 0.2372 |
| **+ Rocchio + neural reranker** | 0.2617 | 0.2571 | **0.2594** |
| ablation: trained network only (no priors) | 0.2343 | 0.2238 | 0.2291 |
| ablation: raw cosine similarities instead of the trained network | 0.2457 | 0.2361 | 0.2409 |

Weights were tuned on the first 400 turns of the first half (only the Rocchio weight changed from the default).
Reading it honestly:
- Rocchio gives most of the gain over Part 1 (+15%).
- The neural reranker, which replaces the BM25 term, matches the BM25-based scoring when both are combined with
  Rocchio (0.2594 against 0.2580, within noise) and beats Part 1 scoring on its own (+5%). So the pipeline is
  neural without losing accuracy; the reranker is not a large accuracy gain by itself in this test.
- Raw cosine similarity without the trained network is competitive here (0.2409). The network is trained on the
  next-track task over the whole catalog (held-out nDCG@20 0.213 on its own, recall@20 0.46), mixing it with the
  raw cosines did not help (nine combinations, 0.249 to 0.260).
- The priors help (0.2372 against 0.2291).

### B) Rejecting a tag or an artist the target track does not have (883 tag and 400 artist scenarios)
Share of the top 5 that carries the rejected attribute, before -> after:
| | tag | artist |
|---|---|---|
| Rocchio only | 0.85 -> 0.71 (cut 17%) | 0.80 -> 0.69 (cut 13%) |
| neural reranker, no constraint | 0.85 -> 0.64 (cut 25%) | 0.80 -> 0.70 (cut 13%) |
| Rocchio + neural, no constraint | 0.85 -> 0.58 (cut 31%) | 0.80 -> 0.66 (cut 18%) |
| Rocchio + neural, penalty x3, no constraint | 0.85 -> 0.46 (cut 45%) | 0.80 -> 0.66 (cut 18%) |
| **Rocchio + neural + constraint** | **0.85 -> 0.00 (cut 100%)** | **0.80 -> 0.00 (cut 100%)** |

The soft neural penalty alone reaches the promised 30% for tags but not for artists; the explicit "no more X"
constraint makes it complete. The scenarios pick attributes the target track does not have, so nDCG rises when the
attribute is removed (0.193 -> 0.303 for tags); this is by construction and shows only that relevance is not hurt.

### C) "More of style X" in chat conditions (400 scenarios; the retrieval query is the profile only)
| | share of top 5 with the style | nDCG@20 |
|---|---|---|
| style not mentioned | 0.13 | 0.247 |
| style in the retrieval query only | 0.22 | 0.292 |
| + style prototype in the trained space (default, weight 0.5) | 0.25 | 0.305 |
| + style prototype in the metadata space | 0.23 | 0.282 |
| style prototype, weight 2 | 0.35 | 0.243 |

Embedding the style *word* with the text encoder hurt (nDCG 0.234 against 0.277 in an early test), which is why
styles are represented by the tracks that carry them.

### D) Diversity (500 turns, 1 - mean pairwise cosine of the top 5, nDCG@20)
| lambda | diversity | nDCG@20 |
|---|---|---|
| none | 0.265 | 0.2738 |
| 0.8 (default for refined lists) | 0.269 | 0.2738 |
| 0.6 | 0.277 | 0.2714 |
| 0.4 | 0.302 | 0.2648 |

"More varied" uses lambda 0.3 over a pool of 200 candidates.

### E) Understanding feedback (`python -m scripts.eval_feedback`)
30 labelled sentences about a shown list: 30/30 correct, median 0.7 s. Five sentences are counted through
equivalent readings (for example "the third and fourth are perfect" as a selection, which adds and so likes the
songs); strictly 25/30. The router overall: 43/43 on `python -m scripts.eval_router`.

## Latency (real LLM, real embeddings, laptop CPU)
A feedback turn makes two LLM calls (router, then one sentence): 1.0 to 2.9 s; other turns 0.2 to 1.1 s. The
server needs about 40-60 s to start (term index, embeddings, trained network, generator).

## Limits
- Accuracy gains come mostly from Rocchio; the neural reranker mainly replaces BM25 scoring at equal accuracy and
  carries the feedback (styles, wishes, penalties) in the embedding space.
- Rejections are only fully effective through the constraint stage.
- The offline implicit-feedback test cannot measure explicit dislikes or wishes; sections B and C use constructed
  scenarios, not real users.
- The first run on a new machine needs the embedding caches (about 0.8 GB download) and `data/reranker.pt`
  (committed; retrain with `python -m scripts.train_reranker`).

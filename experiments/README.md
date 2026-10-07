# Part 1 experiments: how to reproduce

Every number in `docs/part1_experiment_log.md` and in the G1 report comes from one of the scripts below.
Raw (cleaned) outputs from the original runs are in `results/`.

## Setup

```powershell
conda activate musiccrs        # Python 3.11, packages from requirements.txt + sentence-transformers
cd <repo root>                 # scripts also work from any folder (they locate the repo themselves)
mkdir outputs                  # git-ignored scratch folder
```

Datasets are downloaded automatically from Hugging Face on first use (metadata ~18 MB, dialogues ~95 MB).

## Step 0: baseline and ground truth (starter-kit commands, no script of ours)

```powershell
python -m retrieval.evaluation.make_ground_truth --split test --output outputs/ground_truth.json
python -m retrieval.run_bm25_baseline --split test --topk 20 --output outputs/bm25_predictions.json
python -m retrieval.evaluation.evaluate --predictions outputs/bm25_predictions.json --ground_truth outputs/ground_truth.json --catalog_size 47071
```
Result (`results/baseline_eval.txt`): nDCG@20 0.0830, diversity 0.3908, final 0.1446 (published: 0.1444).
Scripts 05-08 need `outputs/ground_truth.json` from the first command.

## Scripts (run in this order)

| script | what it does | output in `results/` | approx. time |
|---|---|---|---|
| `retrieval/analysis.py` (run as `python -m retrieval.analysis`) | train-split statistics: gold repeats in history (0.0%), unique golds (43,597 / 47,071), gold popularity vs catalog | `analysis_train_stats.txt` | ~1 min |
| `00_explore_data.py` | prints the structure of the dialogue / metadata / embedding datasets (columns, example session). Output was read interactively, not saved; streams the 790 MB embeddings file, so it can be slow. | - | minutes |
| `01_artist_continuity.py` | how often the gold track's artist / album appeared earlier in the session (60.7% / 41.6% for turn >= 2) | `01_artist_continuity.txt` | ~1 min |
| `02_bm25_query_variants.py` | 8 BM25 variants (query form, fields, played-track filter) scored with `quick_eval` on a 2,000-turn train sample (seed 0) | `02_bm25_query_variants.txt` | ~10 min |
| `03_feature_weights_grid.py` | weighted rescoring (BM25 + artist + album + popularity + optional current-message BM25), grid search on one 1,000-turn train sample, checked on another | `03_feature_weights_grid.txt` | ~5 min |
| `04_weights_and_diversity_8k.py` | same on an 8,000-turn train dev set (1,000 whole sessions, same size as test), adds the "last-artist" feature, tunes weights on half 1 / checks on half 2, then sweeps the usage-penalty diversity pass (lam, head protection p) | `04_weights_and_diversity_8k.txt` | ~10 min |
| `05_test_check_prototype.py` | first test-split check of the prototype pipeline with weights fixed from script 04 (lam in 0, 0.25, 0.5, 1.0) | `05_test_check_prototype.txt` | ~5 min |
| `06_verify_run_final.py` | scores the repo implementation `retrieval/run_final.py` on test in memory (does not write predictions.json) | `06_verify_run_final.txt` | ~3 min |
| `07_verify_K_and_lam.py` | repo implementation: candidate pool K in {300, 1000} x lam in {0.5, 1.0, 2.0} on test | `07_verify_K_and_lam.txt` | ~8 min |
| `08_hub_analysis.py` | repo implementation with the penalty off: how often tracks are recommended vs how often they are correct (the "hubness" evidence), plus K=1000 numbers for lam 0 and 1 | `08_hub_analysis.txt` | ~4 min |

## Producing the submission file

```powershell
python -m retrieval.run_final     # writes outputs/predictions.json (K=1000, lam=1.0, CPU, deterministic, seed 0)
python -m retrieval.evaluation.evaluate --predictions outputs/predictions.json --ground_truth outputs/ground_truth.json --catalog_size 47071
```
Expected: nDCG@20 0.1755, diversity 0.7263, final 0.2857 (nDCG@1 0.0909, nDCG@10 0.1619).

## Notes

- Tuning rule: feature weights and lam were chosen on train-derived data (scripts 03, 04). The test split was used for evaluation (scripts 05-08); K (300 vs 1000) and small lam sweeps were compared on test.
- Scripts 06-08 import `retrieval.run_final` and override its `K` at runtime; they do not modify any file.
- Shared code: `retrieval/turns.py` (flattens sessions into per-turn records), `retrieval/tune.py` (`quick_eval`), `retrieval/experiment.py` (the first BM25 experiment).
- Times are rough estimates from the original runs on a CPU-only laptop, not benchmarks.

# Experiment registry

## E1 — FEVER English claim-only baseline

- **Research questions:** software/baseline component of RQ1; English context for RQ2; linear explanation and error analysis for RQ4/RQ6.
- **Dataset/task:** official FEVER English labels, claim-only; internal canonical labels preserve the source mapping.
- **Data:** 145,449 source training rows; three exact normalized groups (three rows) overlap the labelled development set and are removed from training. Validation is 20,780 rows from a seven-fold, seeded StratifiedGroupKFold. Final training uses the remaining 145,446 source training rows. Held-out labelled development contains 19,998 rows, balanced at 6,666 per class.
- **Candidates:** word unigrams/bigrams vs character 3–5-grams. Both use TF-IDF, max 150,000 features, `min_df=2`, Logistic Regression `C=2.0`, balanced class weights, `max_iter=1500`, seed 42.
- **Selection:** validation macro F1; word n-grams selected (0.5814 vs 0.5683).
- **Final held-out:** accuracy 0.5509, macro F1 0.5516, weighted F1 0.5516. Per-class metrics and confusion matrix are in `results/fever/latest/metrics.json`.
- **Probability diagnostics:** Brier 0.5686; top-label ECE (10 bins) 0.1083. No post-hoc calibration was applied.
- **Resources:** final fit 12.60 seconds; 19,998 predictions 0.186 seconds; Python 3.14.6, scikit-learn 1.9.1, NumPy 2.5.4, arm64 macOS.
- **Interpretation:** NOT ENOUGH INFORMATION had the lowest class F1 (0.4859). The benchmark score is not real-world fact-checking performance. No causal or manual error-category conclusions are drawn.

## E2 — AfrIFact multilingual claim classification

- **Research questions:** RQ1/RQ2/RQ4/RQ5/RQ6.
- **Status:** blocked on account-level acceptance of the AfrIFact Hugging Face access conditions and subsequent schema/split inspection. The code does not bypass access controls.
- **Next required action:** accept conditions on the official dataset page, authenticate locally, run `scripts/inspect_afrifact.py`, then set verified configuration, split, field mapping, and (if provided) parallel group ID.
- **No results are available.** Dataset-card counts are not treated as local split counts.

## E3 — XLM-R fine-tuning

- **Research questions:** RQ1/RQ2/RQ5.
- **Status:** complete configurable training/evaluation code exists; no task-fine-tuned checkpoint is bundled or claimed. The pretrained base model is not a task classifier.
- **Comparison rule:** display a direct model comparison only if dataset/task, label inventory, and the hash of held-out sample IDs match. Otherwise report runs separately.

## E4 — Evidence ablation

- **Research question:** RQ3.
- **Status:** not run. FEVER's released JSONL provides evidence annotation identifiers, not passage text. AfrIFact must be inspected for actual passage text and provenance before a claim-only vs claim-plus-evidence comparison is defined.

## E5 — Cross-lingual transfer and manual errors

- **Research questions:** RQ5/RQ6.
- **Status:** pending a target dataset with verified language and parallel-group metadata. Exact normalized duplicate protection cannot detect every translated claim. Manual error categories require a documented, inspected sample and are not auto-generated.

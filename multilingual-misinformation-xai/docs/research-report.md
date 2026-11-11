# Multilingual Misinformation Detection with Explainable AI

## Abstract

This project investigates reproducible claim classification and explainability for English and low-resource African languages. AfrIFact is the intended primary benchmark because its published scope includes English, Yoruba, Hausa, and Igbo alongside evidence-oriented tasks. Its files require acceptance of Hugging Face access conditions, and access has not been granted in the current environment. The executable baseline is therefore a separately identified English-only TF-IDF + Logistic Regression experiment on FEVER. On the official labelled development partition (19,998 claims), the selected baseline achieved 0.5509 accuracy and 0.5516 macro F1. No African-language or transformer performance result is claimed.

## Introduction and related work

Automated claim classification is often confused with fact checking. Reliable evidence-based verification requires evidence retrieval and assessment as well as a classifier. FEVER introduced a large English benchmark with SUPPORTS, REFUTES, and NOT ENOUGH INFO labels (Thorne et al., 2018). X-Fact expanded multilingual fact-checking research to real-world claims across 25 languages, though its published language inventory does not cover this project's four target languages (Gupta & Srikumar, 2021). AfrIFact targets information retrieval, evidence extraction, and verification for ten African languages and English (AfrIFact, 2026).

## Research questions

RQ1 compares a traditional TF-IDF baseline with a fine-tuned multilingual transformer on a common evaluation set. RQ2 studies English, Yoruba, Hausa, and Igbo metrics with their sample counts. RQ3 tests evidence ablation only where evidence provenance supports it. RQ4 examines model-faithful explanations. RQ5 studies cross-lingual transfer without parallel-claim leakage. RQ6 analyzes supported held-out error patterns.

## Data and methods

AfrIFact is selected as the primary target, contingent on authorized access and schema inspection. X-Fact is documented as a secondary comparison source but is not selected for target-language experiments. FEVER serves as an independent English software and baseline benchmark. The official FEVER train file contains 145,449 rows; three normalized claim groups (three rows) also occur in the labelled development partition and were removed from training. A seeded, seven-fold StratifiedGroupKFold uses one fold (20,780 examples) for validation and the other folds (124,666 examples) for candidate fitting. The selected configuration was refit on all 145,446 cleaned training rows. The 19,998-row labelled development set is the final held-out evaluation partition. Its three labels are balanced at 6,666 rows each. The FEVER blind test is not used because it is unlabelled.

The traditional baseline is a scikit-learn Pipeline with TF-IDF and Logistic Regression. It compares word unigrams/bigrams with character 3–5-grams; class weighting is balanced, `C=2.0`, `min_df=2`, and vocabulary size is capped at 150,000. Validation macro F1 selects the candidate. The planned transformer uses fine-tuned XLM-RoBERTa Base. Claim-only and evidence-based inputs are different tasks and will be reported separately. Linear model coefficients and token contributions explain fitted scores, not evidence or causal reasoning.

## Experimental setup and metrics

All project-controlled splits use fixed seeds. We report accuracy, macro and weighted F1, per-class metrics, confusion matrix, sample count, Brier score, top-label ECE, and measured runtime. Per-language results require verified dataset language metadata. Hyperparameters are selected using validation data; the held-out partition is used once for final evaluation. Artifacts record model, dataset hashes, split, environment, and configuration metadata.

## Results, error analysis, and explainability

The word unigram/bigram candidate reached validation macro F1 0.5814, compared with 0.5683 for character 3–5-grams. On labelled FEVER dev, the refit model reached accuracy 0.5509, macro F1 0.5516, and weighted F1 0.5516. Per-class F1 was 0.4859 for NOT ENOUGH INFORMATION, 0.6236 for REFUTED, and 0.5453 for SUPPORTED. The three classes each had 6,666 examples, so macro and weighted F1 coincide. The confusion matrix (rows true; columns predicted, ordered NOT ENOUGH INFORMATION / REFUTED / SUPPORTED) was `[[3167, 1145, 2354], [1373, 3974, 1319], [1830, 960, 3876]]`. Multiclass Brier score was 0.5686 and top-label ECE with ten bins was 0.1083; probabilities were not post-hoc calibrated.

The final FEVER run took 12.60 seconds for fitting and 0.186 seconds for held-out predictions on Python 3.14.6, scikit-learn 1.9.1, NumPy 2.5.4, and arm64 macOS. Timings are from one local run. The confusion matrix shows most frequent errors from NOT ENOUGH INFORMATION into SUPPORTED (2,354) and from SUPPORTED into NOT ENOUGH INFORMATION (1,830). This is an observed confusion pattern, not an explanation of cause. Error categories have not been manually annotated.

AfrIFact and XLM-R metrics are unavailable and are not fabricated. The downloaded FEVER JSONL has evidence annotations but no textual evidence passage; evidence ablation is not run. No cross-lingual conclusion is currently possible.

## Discussion, limitations, and ethics

The gated primary data creates an access prerequisite. Until fulfilled, this codebase demonstrates research plumbing on a clearly separated English benchmark, not multilingual reliability. FEVER's modified-Wikipedia claim style limits ecological validity. Dataset annotations are not universal truth labels. Predictions should not be used to adjudicate claims or moderate users. Dataset terms and original attribution remain mandatory.

## Conclusion

The pipeline separates dataset provenance, claim-only classification, evidence-based verification, and real-world fact checking. The measured English benchmark result is modest and class performance differs; this supports caution about treating a single aggregate as sufficient. Quantitative conclusions about the target African languages still await actual runs on the authorized primary benchmark.

## References

- AfrIFact authors. (2026). *AfrIFact: Cultural Information Retrieval, Evidence Extraction And Fact Checking for African Languages*. EMNLP. https://aclanthology.org/2026.emnlp-main.133/
- Gupta, A., & Srikumar, V. (2021). *X-FACT: A New Benchmark Dataset for Multilingual Fact Checking*. ACL. https://aclanthology.org/2021.acl-short.86/
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL-HLT. https://aclanthology.org/N18-1074/
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. ACL. https://aclanthology.org/2020.acl-main.747/

# Multilingual Misinformation Detection with Explainable AI

## Abstract

This project investigates reproducible claim classification and explainability for English and low-resource African languages. AfrIFact is the intended primary benchmark because its published scope includes English, Yoruba, Hausa, and Igbo alongside evidence-oriented tasks. Its files require acceptance of Hugging Face access conditions, and access has not been granted in the current environment. The executable baseline is therefore a separately identified English-only TF-IDF + Logistic Regression experiment on FEVER. No African-language or transformer performance result is claimed. This report records the research design and current implementation status; measured findings will be added only from completed runs.

## Introduction and related work

Automated claim classification is often confused with fact checking. Reliable evidence-based verification requires evidence retrieval and assessment as well as a classifier. FEVER introduced a large English benchmark with SUPPORTS, REFUTES, and NOT ENOUGH INFO labels (Thorne et al., 2018). X-Fact expanded multilingual fact-checking research to real-world claims across 25 languages, though its published language inventory does not cover this project's four target languages (Gupta & Srikumar, 2021). AfrIFact targets information retrieval, evidence extraction, and verification for ten African languages and English (AfrIFact, 2026).

## Research questions

RQ1 compares a traditional TF-IDF baseline with a fine-tuned multilingual transformer on a common evaluation set. RQ2 studies English, Yoruba, Hausa, and Igbo metrics with their sample counts. RQ3 tests evidence ablation only where evidence provenance supports it. RQ4 examines model-faithful explanations. RQ5 studies cross-lingual transfer without parallel-claim leakage. RQ6 analyzes supported held-out error patterns.

## Data and methods

AfrIFact is selected as the primary target, contingent on authorized access and schema inspection. X-Fact is documented as a secondary comparison source but is not selected for target-language experiments. FEVER serves as an independent English software and baseline benchmark. FEVER's official training set is split into training and validation groups using normalized-claim grouping; its labelled development set is the final held-out evaluation partition. The FEVER blind test is not used because it is unlabelled.

The traditional baseline is a scikit-learn TF-IDF + Logistic Regression Pipeline. The planned transformer uses fine-tuned XLM-RoBERTa Base. Claim-only and evidence-based inputs are different tasks and will be reported separately. Linear model coefficients and token contributions explain fitted scores, not evidence or causal reasoning.

## Experimental setup and metrics

All project-controlled splits use fixed seeds. Report accuracy, macro and weighted F1, per-class metrics, confusion matrix, sample count, and measured runtime. Per-language results require verified dataset language metadata. Hyperparameters are selected using validation data; the held-out partition is used once for final evaluation. Artifacts record model, dataset, split, environment, and configuration metadata.

## Results, error analysis, and explainability

**Pending execution.** AfrIFact and XLM-R metrics are unavailable and are not fabricated. FEVER baseline results will be populated from the reproducible training command after data download and evaluation. Error categories will be assigned only where inspected examples support them. No cross-lingual or evidence-ablation conclusions are currently possible.

## Discussion, limitations, and ethics

The gated primary data creates an access prerequisite. Until fulfilled, this codebase demonstrates research plumbing on a clearly separated English benchmark, not multilingual reliability. FEVER's modified-Wikipedia claim style limits ecological validity. Dataset annotations are not universal truth labels. Predictions should not be used to adjudicate claims or moderate users. Dataset terms and original attribution remain mandatory.

## Conclusion

The research pipeline separates dataset provenance, claim-only classification, evidence-based verification, and real-world fact checking. The current scientific result is a documented access and coverage gap; quantitative conclusions await actual runs on the authorized target benchmark.

## References

- AfrIFact authors. (2026). *AfrIFact: Cultural Information Retrieval, Evidence Extraction And Fact Checking for African Languages*. EMNLP. https://aclanthology.org/2026.emnlp-main.133/
- Gupta, A., & Srikumar, V. (2021). *X-FACT: A New Benchmark Dataset for Multilingual Fact Checking*. ACL. https://aclanthology.org/2021.acl-short.86/
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL-HLT. https://aclanthology.org/N18-1074/
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. ACL. https://aclanthology.org/2020.acl-main.747/

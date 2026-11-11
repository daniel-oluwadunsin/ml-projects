# Research methodology

## Research questions

1. How does TF-IDF + Logistic Regression compare with fine-tuned XLM-R on a shared, valid split?
2. How does performance vary across English, Yoruba, Hausa, and Igbo, with sample sizes shown?
3. Does supplied evidence improve verification when its provenance and split are valid?
4. Which features or tokens affect each model's score, and do errors suggest shortcuts?
5. What cross-lingual transfer is supported by the available data without parallel-claim leakage?
6. What supported error patterns appear in held-out predictions?

## Task definition

Claim-only classification predicts a dataset annotation from claim text. It does not establish factual truth. Evidence-based verification must consume a claim and properly sourced evidence and predict whether that evidence supports, refutes, or is insufficient. Retrieval is a separate component. No external search service or LLM is called by the predictor.

## Planned protocol

AfrIFact is the primary target benchmark, but data access is currently gated. After access, inspect the exact config, feature schema, data provenance, official partitions, parallel claims, and label definitions before selecting an experiment. Preserve source IDs and split identifiers. Group parallel/duplicate claims before splitting where the official split does not already guarantee non-overlap. Fit vectorizers and select hyperparameters on training/validation data only. Keep final evaluation out of model selection.

The available runnable experiment is FEVER English claim-only classification. It uses the official training data, a seeded validation split grouped on normalized claim text, and the separately published labelled development split as a held-out evaluation set. Features are fitted only on the training partition. This experiment is not a substitute for AfrIFact and will be labeled FEVER in all artifacts and UI views.

## Models

The traditional baseline uses a scikit-learn Pipeline containing TF-IDF and multiclass Logistic Regression. The vectorizer learns vocabulary and inverse document frequencies from training examples; the classifier learns a linear score for each class. Its coefficients and per-feature score contributions provide model-faithful explanations. Coefficients indicate model association, not causal importance.

The transformer pipeline fine-tunes `FacebookAI/xlm-roberta-base` for sequence classification. It supports configurable claim-only or claim-plus-evidence inputs, but evidence mode must only run on data with verified evidence fields and provenance. The base checkpoint is not itself a fine-tuned classifier. No transformer score will be displayed until training and evaluation artifacts exist.

## Evaluation

Report accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix, sample count, and measured runtime when available. Macro F1 gives each class equal weight. Also report per-language counts and metrics when actual metadata supports them. A missing language is a data coverage gap, not zero performance. Confidence and calibration are descriptive unless calibration is fit on a separate validation partition.

## Integrity and reproducibility

Seed all project-controlled splits and estimators. Preserve raw labels and provenance. Keep source data and model binaries local. Record dataset identifier, versions, config, split strategy, language coverage, library versions, OS/device, and timestamps in each run. Distinguish published dataset-card facts from locally observed facts and actual model results from proposed experiments.

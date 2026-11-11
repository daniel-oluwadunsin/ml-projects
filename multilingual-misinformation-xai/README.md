# Multilingual Misinformation Detection with Explainable AI

Research project: **Evaluating model reliability across low-resource African languages**.

> A classifier's label is not a verified fact. The runnable baseline is an English-only, claim-only FEVER experiment. AfrIFact is the intended primary benchmark, but its files require accepting Hugging Face access conditions. No Yoruba, Hausa, Igbo, evidence-ablation, or transformer metrics are claimed until the relevant data and experiments are actually available.

## Research questions

1. How does TF-IDF + Logistic Regression compare with fine-tuned XLM-R on the same benchmark?
2. How does performance vary across English, Yoruba, Hausa, and Igbo, and how much test data supports each estimate?
3. Does supplied evidence improve classification when its provenance is known?
4. Which model features influence predictions, and do errors expose shortcuts?
5. Does training transfer across languages without parallel-claim leakage?
6. What patterns are visible in real held-out errors?

## Current status

| Capability | Status |
|---|---|
| Dataset investigation | Source review documented; AfrIFact files gated |
| English baseline | FEVER training/evaluation pipeline being implemented |
| African-language training/evaluation | Pending authorized AfrIFact access |
| XLM-R fine-tuning | Configurable pipeline planned; no fine-tuned model or result yet |
| Explanations | Linear model coefficient and feature-contribution explanations |
| Research dashboard | Shows only available local model/results |

See [dataset investigation](docs/dataset-investigation.md), [methodology](docs/methodology.md), [limitations](docs/limitations.md), and [research report](docs/research-report.md).

## Scope and terminology

The project distinguishes (A) claim-only classification against dataset labels, (B) verification conditioned on supplied evidence, and (C) real-world fact checking that retrieves and judges trustworthy sources. This repository implements A when a trained model is present, prepares a separately configurable evidence-based pathway only after validating evidence fields, and does not implement C. No external LLM or search API participates in predictions.

## Datasets and attribution

- [AfrIFact](https://huggingface.co/datasets/masakhane/AfrIFact), paper: [EMNLP 2026](https://aclanthology.org/2026.emnlp-main.133/). Dataset card says CC BY 4.0 and requires agreeing to access conditions. Follow its acknowledgement and terms. Source data stay local.
- [X-Fact](https://huggingface.co/datasets/utahnlp/x-fact), paper: [ACL 2021](https://aclanthology.org/2021.acl-short.86/). The card lists 25 languages and MIT, but its training and zero-shot language lists do not include English, Yoruba, Hausa, or Igbo; source content may carry additional terms.
- [FEVER](https://fever.ai/dataset/fever.html), paper: [NAACL-HLT 2018](https://aclanthology.org/N18-1074/). English-only benchmark. The official site describes 185,445 claims and labels SUPPORTS, REFUTES, NOT ENOUGH INFO. Its license page applies Wikipedia terms or CC BY-SA 3.0 to annotations. Keep raw data local and respect source terms.

## Setup

Python 3.11–3.14 is configured. Install the research app and development tools:

```bash
cd multilingual-misinformation-xai
uv sync --extra dev
```

Download official FEVER training and labelled development JSONL files from the links on [fever.ai/dataset/fever.html](https://fever.ai/dataset/fever.html), and save them locally as:

```text
data/raw/fever_train.jsonl
data/raw/fever_dev.jsonl
```

Raw data are ignored by Git. Prepare, train, and evaluate:

```bash
uv run python scripts/prepare_data.py --dataset fever
uv run python scripts/run_baseline.py --config configs/baseline.yaml
uv run python scripts/evaluate_models.py --run results/fever/latest
```

The exact `--help` options are authoritative as the scripts are completed. The final held-out split is FEVER's labelled development file; it is not the blind test set.

## AfrIFact access

1. Sign in on the [AfrIFact dataset page](https://huggingface.co/datasets/masakhane/AfrIFact) and accept its terms.
2. Run `hf auth login` locally. Do not paste tokens into source files or commit them.
3. Inspect the real configs/features with `python scripts/inspect_afrifact.py`.
4. Set the actual field mapping in an untracked local config only after inspection.

The adapter will fail with an actionable message if authorization, configuration, or schema is missing. The public card's examples mention a different repository owner; this project will not silently switch repositories. Published counts in docs are card metadata, not observed local rows.

## Baseline concepts

**TF-IDF** represents text with weighted word/ngram counts. Common terms receive less weight; terms more distinctive to a document receive more. **Logistic Regression** learns class-specific linear scores from those features and converts the scores to probabilities with softmax. It is a strong, interpretable baseline, but it can memorize names, topics, and annotation artifacts. It does not retrieve evidence.

The model is a scikit-learn `Pipeline`: vectorizer fitting is restricted to training rows, so validation/test text does not affect the vocabulary or inverse-document frequencies. Unicode is normalized with NFC; diacritics are preserved. Duplicate normalized claims are grouped across the internal train/validation split.

## Transformer

The planned second model fine-tunes [XLM-RoBERTa Base](https://huggingface.co/FacebookAI/xlm-roberta-base), a multilingual pretrained encoder. A pretrained encoder is not a fact-verification model; a task-specific head must be trained on labeled examples. Install heavy dependencies only when needed:

```bash
uv sync --extra dev --extra transformer
uv run python scripts/run_transformer.py --config configs/transformer.yaml
```

Fine-tuning may require substantial memory and accelerator time. The baseline-only app remains available without a transformer artifact. No transformer metrics are displayed until a real run saves them.

## Explainability and evaluation

For Logistic Regression, global views use learned class coefficients. A local explanation multiplies each active TF-IDF feature by its coefficient for the selected class score, then ranks positive/negative contributions. This explains the fitted linear score; it does not reveal causal reasoning or justify a claim. The app displays model label, dataset, task mode, provenance, and uncertainty caveats alongside predictions.

Metrics include accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrices, and sample counts. Per-language metrics require actual language metadata. The validation partition selects model settings; the held-out evaluation partition is used for final reporting. See [methodology](docs/methodology.md).

## Dashboard

Launch with:

```bash
uv run streamlit run app/main.py
```

The multipage dashboard contains overview, claim analysis, model comparison, language performance, explainability, error analysis, dataset explorer, methodology, and references. Pages show clear empty states when the local model or results do not exist. Claim input is limited to 2,000 characters and is treated as untrusted text.

## Reproducibility and tests

Seed, dataset/split identifiers, model configuration, package versions, run metadata, and machine-readable predictions/results are recorded. Raw data and model binaries are excluded; compact metrics and documentation belong in Git.

```bash
uv run pytest
uv run ruff check .
```

## Limitations and ethics

An English FEVER result cannot demonstrate Yoruba, Hausa, or Igbo performance. Dataset labels encode annotation decisions, not universal truth. FEVER's edited-Wikipedia claims differ from current real-world misinformation. Models may exploit dataset artifacts. Do not use this prototype to moderate people or adjudicate real claims. Evidence-based conclusions require relevant, trustworthy evidence; a probability alone is not proof.

## License

Project code is MIT licensed. Dataset and pretrained-model terms remain separate and must be respected; see the dataset investigation document and original sources.

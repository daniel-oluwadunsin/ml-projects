# Dataset investigation

**Status:** source review completed 2026-11-11. AfrIFact file access is gated and has not been granted in this environment. Counts below are published source metadata, not locally downloaded or independently recomputed statistics.

## Comparison

| Dataset | Languages and size | Task, schema, labels | Splits/access | License and suitability |
|---|---|---|---|---|
| [AfrIFact](https://huggingface.co/datasets/masakhane/AfrIFact), [paper](https://aclanthology.org/2026.emnlp-main.133/) | 10 African languages plus English. Dataset card reports 9,328 Health and 8,928 Culture examples. Health lists 848 examples per language, including English, Yoruba, Hausa, and Igbo. Culture counts vary by language. | Benchmark spans retrieval, evidence extraction, and verification. Card distribution uses `SUPPORT`, `REFUTES`, `NEI`. The exact loaded features/configuration and field-level provenance must be inspected after access; no schema is assumed by this project. | Hugging Face requires signing in and accepting its access conditions. No local split counts have been verified. | CC BY 4.0 per card, subject to its access conditions and required acknowledgement. Primary dataset because its language and evidence tasks align with the study. |
| [X-Fact](https://huggingface.co/datasets/utahnlp/x-fact), [paper](https://aclanthology.org/2021.acl-short.86/) | 25 languages, 31,189 claims per card. Train 19,079; dev 2,535; in-domain test 3,826; OOD 2,368; zero-shot 3,381. | Real-world claims with five retrieved evidence-document fields and a seven-valued label field. | Public dataset card reports 13 train languages and 12 zero-shot-only languages. Published list includes neither English nor Yoruba, Hausa, or Igbo. | Dataset card identifies MIT; underlying source material may have additional terms. Useful comparative source only if later task/label suitability is established. Not selected for target-language results. |
| [FEVER](https://fever.ai/dataset/fever.html), [paper](https://aclanthology.org/N18-1074/) | 185,445 English claims according to project site. | `id`, `label`, `claim`, `evidence`; labels `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`. Evidence sets are provided for supports/refutes; NEI has a null evidence marker. | Official JSONL train and labelled development downloads. This project further separates a validation subset from train and uses labelled dev as final test. | FEVER license page describes annotations under applicable Wikipedia terms or CC BY-SA 3.0 otherwise. Raw records stay local. Selected only for a separately named English baseline and UI integration; never treated as African-language evidence. |

## Verified source details and caveats

The AfrIFact card says it covers information retrieval, evidence extraction, and fact-checking in ten African languages and English, and lists the three label counts for the Health configuration. It requires accepting access conditions before files can be read. The card includes example loading snippets that name a different repository owner; this project therefore uses the canonical `masakhane/AfrIFact` repository from the source link and requires explicit configuration/column inspection after authorized access. We will not infer an API schema from a distribution table.

X-Fact's published split/language inventory is not aligned with the four target languages. Its card metadata exposes `language`, `claim`, five evidence fields, and a label field, but its label inventory is dataset-specific and must not be merged into another benchmark without a documented mapping.

FEVER claims were produced by altering Wikipedia sentences. The resulting English claim-only classifier measures association with FEVER's annotation process and claim distribution. It does not verify real-world truth, retrieve evidence, or establish multilingual reliability. Its official test is blind/unlabelled; this implementation uses the official labelled development set as the reported held-out evaluation set and does not call it FEVER's blind test.

## Access and selection

1. Visit the [AfrIFact dataset page](https://huggingface.co/datasets/masakhane/AfrIFact), sign in, review and accept the conditions.
2. Install the optional dataset dependency and authenticate locally with `hf auth login` (or a secure `HF_TOKEN` secret). Never commit credentials.
3. Run `python scripts/inspect_afrifact.py` to print the available configurations and feature names. The adapter refuses to guess claim, label, language, or evidence columns.
4. Copy the verified configuration and feature names into a local, untracked config. Download only under the accepted terms. Report actual rows/splits only after inspection.

Until those steps are complete, all African-language evaluation and AfrIFact training are **pending**. FEVER is used only to exercise the runnable English baseline, training/evaluation code, explanations, and dashboard.

## References

- AfrIFact: dataset card and paper linked above; see also the published paper's provenance and annotation discussion before designing evidence experiments.
- Gupta and Srikumar (2021), *X-FACT: A New Benchmark Dataset for Multilingual Fact Checking*, ACL.
- Thorne et al. (2018), *FEVER: a Large-scale Dataset for Fact Extraction and VERification*, NAACL-HLT.

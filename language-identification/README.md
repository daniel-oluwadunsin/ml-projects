# Language Identification

A character-level text classifier that estimates whether a passage is English, French, Spanish, German, or Portuguese. It uses scikit-learn and WiLI-2018; the Streamlit app shows a predicted language and a calibrated confidence estimate for those five supported classes.

## Dataset and attribution

The project uses [WiLI-2018 on Zenodo](https://zenodo.org/records/841984), DOI [10.5281/zenodo.841984](https://doi.org/10.5281/zenodo.841984), created by Martin Thoma. WiLI is published under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/). Cite Martin Thoma, [“The WiLI benchmark dataset for written language identification”](https://arxiv.org/abs/1801.07779). The downloaded archive and extracted source files are not committed.

WiLI provides fixed official train and test files. Each selected language has 500 official training paragraphs and 500 official test paragraphs. Only the five named languages are used in this version.

## Model and preprocessing

The pipeline preserves Unicode text and uses character-level TF-IDF with 2–5 character n-grams, then compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM. The official training subset is divided into a stratified 80/20 training/validation split for model selection. The selected model is chosen by validation macro F1, then accuracy. The official test split remains untouched until final evaluation.

The selected Linear SVM is wrapped in five-fold sigmoid calibration and refit using all 2,500 official training examples. The app’s confidence values come from this calibrated model. The multiclass Brier score is the mean per-example sum of squared differences between the five predicted probabilities and the one-hot true label; lower is better.

## Measured results

Validation comparison:

| Model | Accuracy | Macro F1 | Macro precision | Macro recall |
| --- | ---: | ---: | ---: | ---: |
| Linear SVM | 0.9940 | 0.9940 | 0.9940 | 0.9940 |
| Logistic Regression | 0.9880 | 0.9880 | 0.9882 | 0.9880 |
| Multinomial Naive Bayes | 0.9800 | 0.9800 | 0.9809 | 0.9800 |

On the official 2,500-example test split, the calibrated Linear SVM achieved **accuracy 98.96%**, **macro F1 0.9896**, **macro precision 0.9897**, **macro recall 0.9896**, and **multiclass Brier score 0.0177**. The committed confusion matrix and calibration plot are in `results/test_evaluation.png`; detailed metrics and per-class results are also saved under `results/`.

The local calibrated model is saved at `models/best_pipeline.joblib` and gitignored. Run the training command below to download the data and recreate it.

## Install and run

From this folder, with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python data/download_data.py
python inspect_data.py
python train.py
python evaluate.py
python -m pytest -q
streamlit run app.py
```

`train.py` compares the candidate models, selects by validation macro F1, performs five-fold sigmoid calibration, evaluates once on the untouched official test split, saves the model, and writes results. `evaluate.py` recomputes the final metrics. If the app reports a missing model, run `python train.py` first.

## Limitations and future languages

- The app supports only the five selected languages. Unsupported languages are still assigned to one of these classes, so confidence can be misleading.
- WiLI consists of Wikipedia paragraphs, not chats or social posts. Short strings, code-switching, URLs, names, spelling variation, and informal writing can reduce reliability.
- Confidence is calibrated on this dataset and class set; it is not a guarantee for every source of text.
- Yoruba, Hausa, Igbo, and other languages can be added by extending the language-code mapping and including appropriately licensed data for both training and evaluation.

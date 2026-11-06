# News Article Classification

A beginner-friendly four-class news topic classifier with a Streamlit interface. Enter a headline, article body, or both to estimate whether the story is World, Sports, Business, or Sci/Tech. The app displays calibrated class probabilities as estimates, not guarantees.

## Dataset and attribution

The project uses the standard AG News topic classification split: 120,000 official training examples and 7,600 official test examples across four balanced categories. The downloader retrieves the CSV files from the [widely used AG News source mirror](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv) and verifies the published file checksums.

The original AG corpus description says the data are provided for research and other non-commercial activity. This project does not redistribute raw data or model binaries; the source files are downloaded locally. Cite Xiang Zhang, Junbo Zhao, and Yann LeCun, [“Character-level Convolutional Networks for Text Classification”](https://arxiv.org/abs/1509.01626), and review the [dataset source description](https://github.com/tensorflow/datasets/blob/master/docs/catalog/ag_news_subset.md) before any commercial use.

## Preprocessing and model selection

Headlines and article descriptions are combined. The scikit-learn TF-IDF vectorizer lowercases text, normalizes accents, removes English stop words, and creates word unigrams and bigrams. Its vocabulary is fitted only on the training partition.

The official training set is split into 96,000 training and 24,000 validation examples with stratification. Multinomial Naive Bayes, Logistic Regression, and Linear SVM are compared using validation macro F1, with accuracy as the tie-breaker. The winner is wrapped in five-fold sigmoid calibration and fitted on all 120,000 official training examples. Only after fitting is the untouched official test set used for final evaluation.

## Measured results

Validation comparison:

| Model | Accuracy | Macro F1 | Macro precision | Macro recall |
| --- | ---: | ---: | ---: | ---: |
| Linear SVM | 0.9260 | 0.9259 | 0.9260 | 0.9260 |
| Multinomial Naive Bayes | 0.9174 | 0.9172 | 0.9173 | 0.9174 |
| Logistic Regression | 0.8791 | 0.8788 | 0.8786 | 0.8791 |

The calibrated Linear SVM achieved **accuracy 92.74%**, **macro F1 0.9272**, **macro precision 0.9272**, **macro recall 0.9274**, and **multiclass Brier score 0.1166** on the official 7,600-example test set. The Brier score is the mean per-example sum of squared differences between the four predicted probabilities and the one-hot true category; lower is better. `results/test_evaluation.png` contains the final confusion matrix and one-vs-rest calibration curves.

The fitted model is stored locally at `models/best_pipeline.joblib` and is excluded from Git. Recreate it with the training command below.

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

`train.py` compares models using only the official training data, selects by validation macro F1, fits the calibrated final pipeline, then opens the official test set for evaluation. It saves the local model and compact results. `evaluate.py` recomputes the official test metrics. The app shows a helpful setup error if the model has not been trained.

## Limitations

- A single broad category may not capture a multi-topic article or a story that fits several labels.
- Results reflect this benchmark's source mix and time period. Short headlines, new terminology, and topics outside the four labels can be misclassified.
- Calibration reflects the benchmark test distribution and is not a guarantee on future news.
- AG corpus source terms describe research and non-commercial use; check upstream terms before commercial use.

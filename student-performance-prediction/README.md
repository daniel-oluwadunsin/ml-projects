# Student Performance Prediction

A beginner-friendly regression project that estimates a student's final Mathematics grade (`G3`) from demographic, family, school, study, and lifestyle fields. Earlier period grades `G1` and `G2` are intentionally excluded. Predictions are estimates, not guaranteed outcomes.

## Dataset and attribution

The project uses `student-mat.csv` from Paulo Cortez's [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student%2Bperformance), DOI [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). The dataset is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The downloader places the archive under `data/raw/`; raw data are not committed to this repository.

The mathematics records contain 395 students and 33 source columns. The target grade ranges from 0 to 20. The initial data check found no missing cells. `G1` and `G2` have strong correlations with `G3` in this data (0.801 and 0.905), which is why they are excluded from the model.

## Models and preprocessing

The training pipeline compares Linear Regression, Decision Tree Regressor, and Random Forest Regressor. Categorical fields (including coded education and rating categories) are one-hot encoded; age and absences are scaled. Preprocessing is fitted inside each model pipeline using training data only.

A fixed random seed creates 276 training, 59 validation, and 60 test records. The model with the lowest validation MAE is selected (ties use lower RMSE, then higher R²). The selected estimator is refit on train plus validation data, and the held-out test split is used for final metrics and permutation importance.

## Measured results

Validation comparison (lower MAE/RMSE is better; higher R² is better):

| Model | Validation MAE | Validation RMSE | Validation R² |
| --- | ---: | ---: | ---: |
| Random Forest | 3.262 | 4.103 | 0.297 |
| Decision Tree | 3.373 | 4.250 | 0.246 |
| Linear Regression | 4.001 | 5.096 | -0.084 |

Random Forest was selected. On the held-out test split it achieved **MAE 3.029**, **RMSE 3.781**, and **R² 0.286**. These values are produced by the committed training pipeline and are not a guarantee of performance on other cohorts.

The largest test-set permutation-importance increases in MAE were `failures` (0.572), `absences` (0.387), and `sex` (0.128). Importance describes this fitted model and dataset; it does not establish causation.

Charts and machine-readable outputs are in `results/`: validation comparison, held-out predictions, test metrics, and permutation importance. The saved model is local at `models/best_pipeline.joblib` and is gitignored; rerun training to regenerate it.

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

`train.py` downloads the dataset if needed, compares all three regressors, saves the selected preprocessing/model pipeline, and writes the final test results. `evaluate.py` recomputes the metrics on the same deterministic held-out split. If the app reports that a model is missing, run `python train.py` first.

## Limitations and future work

- The dataset is small and represents two Portuguese schools; its patterns may not transfer to other places or cohorts.
- It is observational data. Feature importance does not imply that changing a feature will change a student's grade.
- The estimator excludes earlier grades by design, which makes the prediction more useful before those results exist but harder than using prior exam grades.
- Future work could compare grouped or repeated splits, add uncertainty intervals, and validate on newer or geographically broader student data.

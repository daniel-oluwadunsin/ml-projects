"""Streamlit interface for the UCI Student Performance model."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from joblib import load
from student_pipeline import (
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    PROJECT_DIR,
)

MODEL_PATH = PROJECT_DIR / "models" / "best_pipeline.joblib"
METRICS_PATH = PROJECT_DIR / "results" / "test_metrics.json"
OPTIONS_PATH = PROJECT_DIR / "results" / "feature_options.json"
RESULTS_DIR = PROJECT_DIR / "results"

st.set_page_config(page_title="Student Grade Forecast", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      .block-container { max-width: 1180px; padding-top: 2rem; }
      .hero { background: linear-gradient(120deg,#132c4b,#246b76); padding: 2rem 2.2rem;
              border-radius: 20px; color: white; margin-bottom: 1.2rem; }
      .hero-kicker { color: #a9ddd4; text-transform: uppercase; letter-spacing: .16em;
                     font-size: .76rem; font-weight: 700; }
      .hero h1 { color: white; margin: .35rem 0 .3rem; font-size: 2.3rem; }
      .hero p { color: #e2edf0; margin-bottom: 0; max-width: 760px; }
      .note { background: #fff7dd; border-left: 4px solid #e1a92b; padding: .9rem 1rem;
              border-radius: 8px; color: #554214; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #e7ebf2;
                                    border-radius: 14px; padding: 1rem; }
    </style>
    <div class="hero">
      <div class="hero-kicker">Student outcomes · regression</div>
      <h1>Student Performance Forecast</h1>
      <p>Explore how student and family characteristics relate to final Mathematics grades.
         Enter a profile to get a model-based estimate on the 0–20 scale.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not OPTIONS_PATH.exists() or not METRICS_PATH.exists():
    st.error(
        "The trained model or its metadata is missing. From this project folder, run `python train.py` first."
    )
    st.stop()


@st.cache_resource
def get_model():
    return load(MODEL_PATH)


@st.cache_data
def get_metadata():
    options = json.loads(OPTIONS_PATH.read_text())
    metrics = json.loads(METRICS_PATH.read_text())
    return options, metrics


try:
    model = get_model()
    feature_options, test_metrics = get_metadata()
except Exception as exc:  # Present a readable setup error rather than a traceback.
    st.error(f"Could not load the saved model files: {exc}")
    st.stop()

st.markdown(
    '<div class="note"><strong>Use responsibly:</strong> this is an estimate for learning and exploration, '
    "not a guaranteed outcome or a basis for decisions about an individual student. Earlier exam grades "
    "G1 and G2 are intentionally excluded.</div>",
    unsafe_allow_html=True,
)


def option_label(column: str, value: object) -> str:
    descriptions = {
        "school": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"},
        "sex": {"F": "Female", "M": "Male"},
        "address": {"U": "Urban", "R": "Rural"},
        "famsize": {"LE3": "3 or fewer members", "GT3": "More than 3 members"},
        "Pstatus": {"A": "Parents apart", "T": "Parents together"},
        "Medu": {
            0: "None",
            1: "Primary",
            2: "Grades 5–9",
            3: "Secondary",
            4: "Higher education",
        },
        "Fedu": {
            0: "None",
            1: "Primary",
            2: "Grades 5–9",
            3: "Secondary",
            4: "Higher education",
        },
        "Mjob": {
            "at_home": "At home",
            "health": "Health",
            "services": "Services",
            "teacher": "Teacher",
            "other": "Other",
        },
        "Fjob": {
            "at_home": "At home",
            "health": "Health",
            "services": "Services",
            "teacher": "Teacher",
            "other": "Other",
        },
        "reason": {
            "home": "Home",
            "reputation": "Reputation",
            "course": "Course",
            "other": "Other",
        },
        "guardian": {"mother": "Mother", "father": "Father", "other": "Other"},
        "traveltime": {
            1: "Under 15 min",
            2: "15–30 min",
            3: "30–60 min",
            4: "Over 60 min",
        },
        "studytime": {
            1: "Under 2 hours",
            2: "2–5 hours",
            3: "5–10 hours",
            4: "Over 10 hours",
        },
        "failures": {0: "None", 1: "One", 2: "Two", 3: "Three or more"},
    }
    detail = descriptions.get(column, {}).get(value)
    if detail:
        return f"{value} · {detail}"
    if column in {"famrel", "freetime", "goout", "Dalc", "Walc", "health"}:
        return f"{value} · scale 1–5"
    return str(value).replace("_", " ").title()


with st.container(border=True):
    st.subheader("Student profile")
    st.caption(
        "The form mirrors the UCI dataset fields. You can use observed dataset categories for categorical fields."
    )
    with st.form("student_profile"):
        left, middle, right = st.columns(3)
        inputs: dict[str, object] = {}
        with left:
            st.markdown("**Background**")
            for column in [
                "school",
                "sex",
                "age",
                "address",
                "famsize",
                "Pstatus",
                "Medu",
                "Fedu",
            ]:
                if column in NUMERIC_COLUMNS:
                    inputs[column] = st.number_input(
                        "Age", min_value=15, max_value=22, value=17, step=1
                    )
                else:
                    inputs[column] = st.selectbox(
                        column,
                        feature_options[column],
                        format_func=lambda value, name=column: option_label(
                            name, value
                        ),
                    )
        with middle:
            st.markdown("**Family & school support**")
            for column in [
                "Mjob",
                "Fjob",
                "reason",
                "guardian",
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic",
            ]:
                inputs[column] = st.selectbox(
                    column,
                    feature_options[column],
                    format_func=lambda value, name=column: option_label(name, value),
                )
        with right:
            st.markdown("**Study & daily life**")
            for column in [
                "traveltime",
                "studytime",
                "failures",
                "famrel",
                "freetime",
                "goout",
                "Dalc",
                "Walc",
                "health",
                "absences",
            ]:
                if column == "absences":
                    inputs[column] = st.number_input(
                        "School absences", min_value=0, max_value=100, value=4, step=1
                    )
                else:
                    inputs[column] = st.selectbox(
                        column,
                        feature_options[column],
                        format_func=lambda value, name=column: option_label(
                            name, value
                        ),
                    )
        submitted = st.form_submit_button(
            "Estimate final grade", type="primary", width="stretch"
        )

if submitted:
    try:
        row = pd.DataFrame(
            [{column: inputs[column] for column in FEATURE_COLUMNS}],
            columns=FEATURE_COLUMNS,
        )
        grade = float(model.predict(row)[0])
        grade = min(20.0, max(0.0, grade))
        st.session_state["grade_prediction"] = grade
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Please check the profile values and try again. Details: {exc}")

if "grade_prediction" in st.session_state:
    predicted = st.session_state["grade_prediction"]
    st.success(f"Estimated final Mathematics grade: **{predicted:.1f} / 20**")
    st.progress(predicted / 20.0, text="Estimated position on the 0–20 grade scale")

tab_performance, tab_factors, tab_about = st.tabs(
    ["Model performance", "Factors", "About & limitations"]
)
with tab_performance:
    st.subheader("Held-out test performance")
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "MAE",
        f"{test_metrics['mae']:.2f} grade points",
        help="Mean absolute error; lower is better.",
    )
    metric_columns[1].metric(
        "RMSE",
        f"{test_metrics['rmse']:.2f} grade points",
        help="Root mean squared error; lower is better.",
    )
    metric_columns[2].metric(
        "R²",
        f"{test_metrics['r2']:.3f}",
        help="Variance explained on the held-out test split.",
    )
    st.caption(
        f"Selected model: {test_metrics['selected_model']} · Test records: {test_metrics['split_sizes']['test']}"
    )
    chart = RESULTS_DIR / "model_results.png"
    if chart.exists():
        st.image(
            str(chart),
            caption="Validation comparison and final held-out test predictions",
            width="stretch",
        )

with tab_factors:
    st.subheader("Which inputs mattered most in this evaluation?")
    st.write(
        "Permutation importance measures how much test-set MAE increases when one input is shuffled. It describes this model and dataset; it does not show causation."
    )
    chart = RESULTS_DIR / "feature_importance.png"
    if chart.exists():
        st.image(str(chart), width="stretch")

with tab_about:
    st.subheader("How this estimate is produced")
    st.write(
        "The model uses the UCI Mathematics records, combines one-hot encoded categorical inputs with scaled numeric inputs, and compares Linear Regression, Decision Tree, and Random Forest regressors. The best validation MAE selected the final model."
    )
    st.write(
        "G1 and G2 are excluded so the model estimates the final grade without earlier exam grades. The held-out test split is reported separately from validation results."
    )
    st.warning(
        "The dataset is small and comes from two Portuguese schools. Patterns may not transfer to other schools, countries, or cohorts. Predictions are estimates, not guaranteed outcomes."
    )
    st.markdown(
        "Dataset attribution: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student%2Bperformance), licensed CC BY 4.0."
    )

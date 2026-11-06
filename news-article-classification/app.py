"""Streamlit interface for the calibrated AG News classifier."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

from news_pipeline import CLASS_LABELS, PROJECT_DIR


MODEL_PATH = PROJECT_DIR / "models" / "best_pipeline.joblib"
METRICS_PATH = PROJECT_DIR / "results" / "test_metrics.json"
st.set_page_config(page_title="News Article Classifier", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      .block-container { max-width: 1160px; padding-top: 2rem; }
      .hero { background: linear-gradient(120deg,#192b46,#476896); padding: 2rem 2.2rem;
              border-radius: 20px; color: white; margin-bottom: 1.2rem; }
      .hero-kicker { color: #c1d2f4; text-transform: uppercase; letter-spacing: .16em;
                     font-size: .76rem; font-weight: 700; }
      .hero h1 { color: white; margin: .35rem 0 .3rem; font-size: 2.35rem; }
      .hero p { color: #e7edf6; margin-bottom: 0; max-width: 760px; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #e7ebf2;
                                    border-radius: 14px; padding: 1rem; }
    </style>
    <div class="hero">
      <div class="hero-kicker">News topics · calibrated text classification</div>
      <h1>News Article Classifier</h1>
      <p>Give a headline, article text, or both. The model estimates whether the story belongs to
         World, Sports, Business, or Sci/Tech.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("The trained model is missing. From this project folder, run `python train.py` first.")
    st.stop()


@st.cache_resource
def get_model():
    return load(MODEL_PATH)


@st.cache_data
def get_metrics():
    return json.loads(METRICS_PATH.read_text())


try:
    model = get_model()
    metrics = get_metrics()
except Exception as exc:
    st.error(f"Could not load the saved model files: {exc}")
    st.stop()

examples = {
    "World": (
        "Regional leaders announce a new cross-border recovery agreement",
        "Representatives from several neighboring countries approved a shared plan to restore transport links and coordinate emergency supplies. The agreement creates a joint committee that will publish progress updates each quarter.",
    ),
    "Sports": (
        "Late goal sends the home side into the final",
        "The winning team scored in the closing minutes after a fast counterattack. Its goalkeeper made several key saves, while the coach praised the players for maintaining their shape under pressure throughout the second half.",
    ),
    "Business": (
        "Retail group reports stronger quarterly revenue",
        "The company said higher online sales and improved inventory management helped lift revenue during the latest quarter. Executives plan to invest in distribution centers while keeping the annual spending target unchanged.",
    ),
    "Sci/Tech": (
        "Researchers demonstrate a more efficient battery material",
        "A university team reported a laboratory process that may improve how quickly energy can move through a battery cell. The researchers said further testing is needed before the material can be evaluated in commercial devices.",
    ),
}

st.caption("Four categories: " + " · ".join(CLASS_LABELS))
sample_category = st.selectbox("Try an illustrative example", CLASS_LABELS)
if st.button("Load example", width="content"):
    st.session_state["news_headline"] = examples[sample_category][0]
    st.session_state["news_body"] = examples[sample_category][1]

with st.form("news_article"):
    headline = st.text_input(
        "Headline",
        value=st.session_state.get("news_headline", ""),
        key="news_headline",
        placeholder="Optional headline",
    )
    article = st.text_area(
        "Article text",
        value=st.session_state.get("news_body", ""),
        key="news_body",
        height=220,
        placeholder="Paste the article text here…",
    )
    submitted = st.form_submit_button("Classify article", type="primary", width="stretch")

if submitted:
    combined = f"{headline.strip()} . {article.strip()}".strip(" .")
    if not headline.strip() and not article.strip():
        st.error("Enter a headline or article text before requesting a prediction.")
    else:
        try:
            category = str(model.predict([combined])[0])
            probabilities = model.predict_proba([combined])[0]
            probability_map = {
                str(label): float(probabilities[index])
                for index, label in enumerate(model.classes_)
            }
            st.session_state["news_prediction"] = {
                "category": category,
                "confidence": probability_map[category],
                "probabilities": probability_map,
                "length": len(combined),
            }
        except (KeyError, ValueError, TypeError) as exc:
            st.error(f"The article could not be classified. Details: {exc}")

if "news_prediction" in st.session_state:
    result = st.session_state["news_prediction"]
    st.success(f"Predicted category: **{result['category']}**")
    confidence_column, notice_column = st.columns([1, 2])
    confidence_column.metric("Calibrated confidence", f"{result['confidence']:.1%}")
    if result["length"] < 80:
        notice_column.warning("This is a short input. Treat its category and confidence cautiously.")
    else:
        notice_column.info("Probabilities are calibration estimates, not a guarantee of correctness.")
    probability_frame = pd.DataFrame(
        {"Estimated probability": result["probabilities"]}
    ).sort_values("Estimated probability", ascending=True)
    st.bar_chart(probability_frame, horizontal=True, x_label="Estimated probability", y_label="Category")

tab_performance, tab_model, tab_limits = st.tabs(
    ["Evaluation", "How it works", "Limitations"]
)
with tab_performance:
    st.subheader("Official test performance")
    first, second, third, fourth = st.columns(4)
    first.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    second.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    third.metric("Macro precision", f"{metrics['macro_precision']:.3f}")
    fourth.metric("Macro recall", f"{metrics['macro_recall']:.3f}")
    st.caption(
        f"Selected model: {metrics['selected_model']} · Test records: {metrics['official_test_rows']} · "
        f"Multiclass Brier score: {metrics['multiclass_brier']:.4f} (lower is better)."
    )
    chart = PROJECT_DIR / "results" / "test_evaluation.png"
    if chart.exists():
        st.image(
            str(chart),
            width="stretch",
            caption="Confusion matrix and calibration curves on AG News's official test set",
        )

with tab_model:
    st.subheader("TF-IDF + calibrated Linear SVM")
    st.write("The project compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM. TF-IDF uses lowercased word unigrams and bigrams; the vocabulary is fitted only on training examples. Validation macro F1 selected the final model.")
    st.write("The final classifier is wrapped in five-fold sigmoid calibration, trained only with AG News's official training data. The official test set is used for the reported final results.")
    st.write("No pretrained language model or external prediction service is used.")

with tab_limits:
    st.subheader("Scope and limitations")
    st.write("The model predicts one of four broad topics from the AG News benchmark. A story may reasonably span multiple categories, while this classifier always chooses one label.")
    st.write("Results reflect the benchmark's writing style and historical distribution. Short headlines, mixed-topic articles, new terminology, and topics outside the four classes can be misclassified. Calibrated probabilities are estimates, not certainty.")
    st.info("The source describes the AG corpus as available for research and non-commercial activity. This repository does not redistribute the raw corpus; check the upstream terms before commercial use.")
    st.caption("Sample text in the app is illustrative demonstration input and is not used for training or evaluation.")

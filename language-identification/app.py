"""Streamlit interface for the five-language WiLI model."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from joblib import load
from language_pipeline import LANGUAGE_NAMES, PROJECT_DIR

MODEL_PATH = PROJECT_DIR / "models" / "best_pipeline.joblib"
METRICS_PATH = PROJECT_DIR / "results" / "test_metrics.json"
st.set_page_config(page_title="Language Identifier", page_icon="🌍", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      .block-container { max-width: 1120px; padding-top: 2rem; }
      .hero { background: linear-gradient(120deg,#182947,#315e8a); padding: 2rem 2.2rem;
              border-radius: 20px; color: white; margin-bottom: 1.2rem; }
      .hero-kicker { color: #a9dce1; text-transform: uppercase; letter-spacing: .16em;
                     font-size: .76rem; font-weight: 700; }
      .hero h1 { color: white; margin: .35rem 0 .3rem; font-size: 2.35rem; }
      .hero p { color: #e2edf0; margin-bottom: 0; max-width: 760px; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #e7ebf2;
                                    border-radius: 14px; padding: 1rem; }
    </style>
    <div class="hero">
      <div class="hero-kicker">Text classification · character patterns</div>
      <h1>Language Identifier</h1>
      <p>Paste a passage to estimate whether it is English, French, Spanish, German, or Portuguese.
         The classifier learns character n-grams from WiLI-2018.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error(
        "The trained model is missing. From this project folder, run `python train.py` first."
    )
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
    "English": "A careful reader can often recognize a language from the small patterns that letters form across a sentence. This example uses ordinary prose with several common English character combinations.",
    "French": "La langue française utilise des accents et des combinaisons de lettres qui lui donnent une apparence particulière. Un paragraphe plus long fournit généralement de meilleurs indices qu'une phrase très courte.",
    "Spanish": "La identificación de idiomas puede aprovechar las secuencias de caracteres que aparecen con frecuencia en cada lengua. Un texto más largo suele ofrecer señales más fiables que unas pocas palabras.",
    "German": "Die automatische Spracherkennung untersucht wiederkehrende Buchstabenfolgen und typische Schreibweisen. Ein längerer Abschnitt liefert meistens bessere Hinweise als ein einzelnes kurzes Wort.",
    "Portuguese": "A identificação de idiomas observa sequências de caracteres que aparecem com frequência em cada língua. Um texto mais longo costuma oferecer pistas mais úteis do que algumas palavras isoladas.",
}

st.markdown("**Languages in this version:** " + " · ".join(LANGUAGE_NAMES.values()))
sample_language = st.selectbox("Try an example", list(examples))
if st.button("Load example", width="content"):
    st.session_state["language_text"] = examples[sample_language]

with st.form("language_input"):
    text = st.text_area(
        "Text to identify",
        value=st.session_state.get("language_text", ""),
        key="language_text",
        height=190,
        placeholder="Paste a sentence or paragraph here…",
        help="Longer passages usually provide stronger character-pattern evidence.",
    )
    submitted = st.form_submit_button(
        "Identify language", type="primary", width="stretch"
    )

if submitted:
    if not text.strip():
        st.error("Enter some text before requesting a prediction.")
    else:
        try:
            predicted_code = str(model.predict([text])[0])
            probabilities = model.predict_proba([text])[0]
            classes = list(model.classes_)
            named_probabilities = {
                LANGUAGE_NAMES[code]: float(probabilities[index])
                for index, code in enumerate(classes)
                if code in LANGUAGE_NAMES
            }
            confidence = named_probabilities[LANGUAGE_NAMES[predicted_code]]
            st.session_state["language_prediction"] = {
                "code": predicted_code,
                "confidence": confidence,
                "probabilities": named_probabilities,
                "length": len(text.strip()),
            }
        except (KeyError, ValueError, TypeError) as exc:
            st.error(f"The text could not be classified. Details: {exc}")

if "language_prediction" in st.session_state:
    result = st.session_state["language_prediction"]
    name = LANGUAGE_NAMES[result["code"]]
    st.success(f"Predicted language: **{name}**")
    confidence_column, notice_column = st.columns([1, 2])
    confidence_column.metric("Calibrated confidence", f"{result['confidence']:.1%}")
    if result["length"] < 80:
        notice_column.warning(
            "This is a short text. Treat its prediction and confidence cautiously."
        )
    else:
        notice_column.info(
            "Confidence is calibrated on the selected WiLI languages; it is not a guarantee."
        )
    probability_frame = pd.DataFrame(
        {"Estimated probability": result["probabilities"]}
    ).sort_values("Estimated probability", ascending=True)
    st.bar_chart(
        probability_frame,
        horizontal=True,
        x_label="Estimated probability",
        y_label="Language",
    )

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
        f"Selected model: {metrics['selected_model']} · Test examples: {metrics['official_test_records']} · "
        f"Multiclass Brier score: {metrics['multiclass_brier']:.4f} (lower is better)."
    )
    chart = PROJECT_DIR / "results" / "test_evaluation.png"
    if chart.exists():
        st.image(
            str(chart),
            width="stretch",
            caption="Confusion matrix and calibration curves on WiLI's official test split",
        )

with tab_model:
    st.subheader("Character-level TF-IDF + calibrated Linear SVM")
    st.write(
        "The pipeline compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM using character n-grams. Validation macro F1 selected the final classifier. The final SVM is calibrated with five-fold sigmoid calibration on the official training subset."
    )
    st.write(
        "The app shows calibrated probabilities for the five supported languages. The model does not contain a pretrained language model or call an external prediction service."
    )

with tab_limits:
    st.subheader("What this model can and cannot tell you")
    st.write(
        "This version predicts only English, French, Spanish, German, and Portuguese. If you enter another language, the model still chooses among those five; its confidence can be misleading for unsupported languages."
    )
    st.write(
        "Very short strings, names, URLs, code, mixed-language text, spelling errors, and informal writing can reduce reliability. WiLI contains Wikipedia passages, which differ from casual conversation."
    )
    st.write(
        "Yoruba, Hausa, Igbo, and other languages can be added by extending the language-code configuration and including appropriately licensed training and evaluation data for each language."
    )
    st.markdown(
        "Dataset: [WiLI-2018 on Zenodo](https://zenodo.org/records/841984), licensed ODbL. Cite Martin Thoma, [The WiLI benchmark dataset for written language identification](https://arxiv.org/abs/1801.07779)."
    )

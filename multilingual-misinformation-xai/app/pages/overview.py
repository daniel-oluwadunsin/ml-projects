from __future__ import annotations

import streamlit as st

from app.ui import METRICS_PATH, load_json

st.title("Multilingual Misinformation Detection with Explainable AI")
st.subheader("A reproducible study of dataset-label classification and model reliability")
st.warning(
    "The available baseline is English-only and predicts FEVER dataset labels from claim text. "
    "It is not a general fact-checker. AfrIFact is the intended African-language benchmark, "
    "but its files require access approval; no African-language metrics are available yet."
)

metrics = load_json(str(METRICS_PATH))
columns = st.columns(4)
if metrics and metrics.get("status") == "measured":
    heldout = metrics["heldout"]
    columns[0].metric("Runnable model", "TF-IDF + Logistic Regression")
    columns[1].metric("Evaluation dataset", "FEVER labelled dev")
    columns[2].metric("Held-out examples", f"{heldout['n_examples']:,}")
    columns[3].metric("Macro F1", f"{heldout['macro_f1']:.3f}")
else:
    columns[0].metric("Runnable model", "Not trained")
    columns[1].metric("AfrIFact", "Access gated")
    columns[2].metric("Transformer", "Not fine-tuned")
    columns[3].metric("Target-language results", "Pending")
    st.info("Train the FEVER baseline to populate this dashboard with measured results.")

st.markdown("### Research questions")
st.markdown(
    "1. How does a TF-IDF baseline compare with fine-tuned XLM-R on the same data?\n"
    "2. How do English, Yoruba, Hausa, and Igbo results vary with sample size and domain?\n"
    "3. Does supplied evidence help when its provenance is verified?\n"
    "4. What text features drive model scores?\n"
    "5. Does cross-lingual transfer survive duplicate and translation leakage checks?\n"
    "6. What do supported held-out errors reveal about dataset and model limitations?"
)

st.markdown("### Dataset and model status")
st.dataframe(
    [
        {
            "Component": "AfrIFact",
            "Status": "Awaiting accepted Hugging Face access",
            "Use": "Primary target benchmark",
        },
        {
            "Component": "X-Fact",
            "Status": "Reviewed; target languages absent from published inventory",
            "Use": "Secondary context only",
        },
        {
            "Component": "FEVER",
            "Status": "Runnable when local files and model exist",
            "Use": "Separate English baseline",
        },
        {
            "Component": "XLM-RoBERTa",
            "Status": "Training pipeline available; no fine-tuned artifact by default",
            "Use": "Future comparable experiment",
        },
    ],
    hide_index=True,
    use_container_width=True,
)

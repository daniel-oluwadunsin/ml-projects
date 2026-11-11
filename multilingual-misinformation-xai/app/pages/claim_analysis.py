from __future__ import annotations

import streamlit as st

from app.ui import (
    MODEL_METADATA_PATH,
    load_baseline,
    load_json,
    show_missing_model,
    show_task_notice,
)
from ml.explainability.linear_explanations import local_feature_contributions

st.title("Claim analysis")
show_task_notice()
st.selectbox("Model language coverage", ["English (FEVER only)"], disabled=True)
st.selectbox("Task mode", ["Claim-only classification"], disabled=True)
st.caption("Evidence-based classification is unavailable because no evidence-trained model exists.")
claim = st.text_area(
    "Enter an English claim",
    max_chars=2_000,
    height=130,
    placeholder="Example: A statement from a published benchmark, not a request for fact checking.",
)

model = load_baseline()
if model is None:
    show_missing_model()
elif st.button("Classify dataset label", type="primary"):
    if not claim.strip():
        st.error("Enter a non-empty claim.")
    else:
        predicted = str(model.predict([claim])[0])
        probabilities = model.predict_proba([claim])[0]
        class_probabilities = sorted(
            zip(model.classes_, probabilities, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        st.subheader(f"Predicted FEVER label: {predicted.replace('_', ' ')}")
        st.caption(
            "This is a model association with an English benchmark label. It is not an evidence "
            "retrieval result and is not a judgment that the claim is true or false."
        )
        st.dataframe(
            [
                {"Dataset label": label.replace("_", " "), "Model probability": float(probability)}
                for label, probability in class_probabilities
            ],
            hide_index=True,
            use_container_width=True,
        )
        explanation = local_feature_contributions(model, claim, predicted, top_n=12)
        st.markdown(f"**Explanation method:** {explanation['method']}")
        st.caption(explanation["warning"])
        st.dataframe(
            explanation["feature_contributions"], hide_index=True, use_container_width=True
        )
        metadata = load_json(str(MODEL_METADATA_PATH))
        if metadata:
            st.caption(
                f"Model provenance: {metadata['dataset']} · trained {metadata['created_at_utc']} · "
                f"selected validation setup: {metadata['selected_candidate']}."
            )
        else:
            st.caption("Model provenance: local FEVER training run; metadata is unavailable.")

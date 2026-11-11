from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui import TRANSFORMER_MODEL_PATH, load_baseline, load_transformer, show_missing_model
from ml.explainability.linear_explanations import global_feature_importance

st.title("Explainability explorer")
st.markdown(
    "Global coefficients describe which TF-IDF features receive high class-specific weights. "
    "They explain the fitted linear scores, not factual truth or causal relationships."
)
model = load_baseline()
if model is None:
    show_missing_model()
    st.stop()

selected_class = st.selectbox("Class score to inspect", list(model.classes_))
top_n = st.slider("Number of features", min_value=5, max_value=30, value=15)
features = global_feature_importance(model, selected_class, top_n=top_n)
for side, title in [
    ("positive", "Features increasing this class score"),
    ("negative", "Features decreasing this class score"),
]:
    st.markdown(f"### {title}")
    frame = pd.DataFrame(features[side])
    st.plotly_chart(
        px.bar(frame, x="coefficient", y="feature", orientation="h"), use_container_width=True
    )
st.caption(
    "Per-example contributions are available on the Claim analysis page after a model prediction."
)

st.divider()
st.markdown("### Fine-tuned XLM-R token attribution")
if not (TRANSFORMER_MODEL_PATH / "config.json").is_file():
    st.info(
        "No local task-fine-tuned XLM-R model is available. The pretrained base encoder is not "
        "used as a classifier."
    )
else:
    text = st.text_area(
        "English or supported-language input for the fine-tuned model",
        max_chars=2_000,
        key="xlmr_explanation_text",
    )
    if st.button("Explain XLM-R class score"):
        if not text.strip():
            st.error("Enter a non-empty input.")
        else:
            try:
                import torch

                tokenizer, transformer = load_transformer()
                encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(
                    next(transformer.parameters()).device
                )
                with torch.no_grad():
                    logits = transformer(**encoded).logits[0]
                    probabilities = torch.softmax(logits, dim=-1)
                labels = {int(key): value for key, value in transformer.config.id2label.items()}
                predicted_index = int(probabilities.argmax().item())
                st.write(f"Predicted dataset label: **{labels[predicted_index]}**")
                st.caption(
                    "Model probabilities are not calibrated confidence and do not establish "
                    "factual correctness."
                )
                from ml.explainability.transformer_explanations import integrated_gradients_tokens

                attributions = integrated_gradients_tokens(
                    transformer, tokenizer, text, target_class=predicted_index
                )
                attribution_frame = pd.DataFrame(attributions)
                attribution_frame["direction"] = attribution_frame["attribution"].map(
                    lambda value: "Raises score" if value > 0 else "Lowers score"
                )
                st.plotly_chart(
                    px.bar(
                        attribution_frame,
                        x="attribution",
                        y="token",
                        color="direction",
                        orientation="h",
                        color_discrete_map={"Raises score": "#087E73", "Lowers score": "#B64C4C"},
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "Integrated Gradients attributes the selected class logit to token embeddings "
                    "relative to a zero baseline. This does not establish "
                    "causal reasoning or truth."
                )
            except Exception as exc:
                st.error(f"Could not produce transformer attribution ({type(exc).__name__}): {exc}")

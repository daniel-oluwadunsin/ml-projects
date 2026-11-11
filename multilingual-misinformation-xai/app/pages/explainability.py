from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui import load_baseline, show_missing_model
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
for side, title in [("positive", "Features increasing this class score"), ("negative", "Features decreasing this class score")]:
    st.markdown(f"### {title}")
    frame = pd.DataFrame(features[side])
    st.plotly_chart(px.bar(frame, x="coefficient", y="feature", orientation="h"), use_container_width=True)
st.caption("Per-example contributions are available on the Claim analysis page after a model prediction.")

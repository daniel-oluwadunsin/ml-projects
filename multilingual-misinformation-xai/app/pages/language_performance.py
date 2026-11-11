from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui import METRICS_PATH, TRANSFORMER_METRICS_PATH, load_json

st.title("Language performance explorer")
metrics = load_json(str(METRICS_PATH))
transformer_metrics = load_json(str(TRANSFORMER_METRICS_PATH))
available = []
if metrics:
    available.append(("FEVER English baseline", metrics))
if transformer_metrics:
    available.append(("AfrIFact XLM-R", transformer_metrics))
if not available:
    st.info("Language metrics appear after a measured evaluation is available.")
    st.stop()

run_name = st.selectbox("Measured run", [name for name, _ in available])
selected_metrics = dict(available)[run_name]
language_metrics = selected_metrics.get("languages", {})
if not language_metrics:
    st.info("This run contains no verified language metadata.")
    st.stop()

frame = pd.DataFrame(
    [
        {
            "Language": language,
            "Examples": values["n_examples"],
            "Accuracy": values["accuracy"],
            "Macro F1": values["macro_f1"],
        }
        for language, values in language_metrics.items()
    ]
)
st.dataframe(frame, hide_index=True, use_container_width=True)
st.plotly_chart(
    px.bar(frame, x="Language", y="Macro F1", color="Examples", range_y=[0, 1]),
    use_container_width=True,
)
st.caption(f"Selected run: {run_name}. Only languages in this run's measured metadata are shown.")
if run_name == "FEVER English baseline":
    st.caption(
        "The current FEVER run covers English only. Yoruba, Hausa, and Igbo results are not "
        "available."
    )
st.info(
    "Differences between language scores require real matched data and context; they do not by "
    "themselves prove language bias."
)

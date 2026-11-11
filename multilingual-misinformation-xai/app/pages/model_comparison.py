from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from app.ui import METRICS_PATH, load_json

st.title("Model comparison")
metrics = load_json(str(METRICS_PATH))
if not metrics:
    st.info("No measured model results are available. Train and evaluate the FEVER baseline first.")
    st.stop()

heldout = metrics["heldout"]
st.caption(f"Dataset: {metrics['dataset']} · task: {metrics['task']} · n={heldout['n_examples']:,}")
candidate_rows = []
for candidate in metrics.get("validation_candidates", []):
    candidate_rows.append({
        "Validation candidate": candidate["name"],
        "Validation macro F1": candidate["validation"]["macro_f1"],
        "Validation accuracy": candidate["validation"]["accuracy"],
        "Selected": candidate["name"] == metrics["selected_candidate"],
    })
if candidate_rows:
    st.markdown("### Validation model selection")
    st.dataframe(pd.DataFrame(candidate_rows), hide_index=True, use_container_width=True)

st.markdown("### Held-out metrics")
summary = pd.DataFrame([{
    "Accuracy": heldout["accuracy"],
    "Macro precision": heldout["macro_precision"],
    "Macro recall": heldout["macro_recall"],
    "Macro F1": heldout["macro_f1"],
    "Weighted F1": heldout["weighted_f1"],
}]).melt(var_name="Metric", value_name="Score")
st.plotly_chart(px.bar(summary, x="Metric", y="Score", range_y=[0, 1]), use_container_width=True)

labels = heldout["labels"]
st.markdown("### Per-class report")
per_class = pd.DataFrame.from_dict(heldout["per_class"], orient="index").rename_axis("Label")
st.dataframe(per_class, use_container_width=True)
st.markdown("### Confusion matrix")
st.plotly_chart(
    ff.create_annotated_heatmap(
        z=heldout["confusion_matrix"], x=[x.replace("_", " ") for x in labels],
        y=[x.replace("_", " ") for x in labels], colorscale="Blues", showscale=True,
    ),
    use_container_width=True,
)
st.info("Transformer comparison is pending a real fine-tuning and evaluation run on an identical dataset split.")

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import (
    METRICS_PATH,
    RUN_METADATA_PATH,
    TRANSFORMER_METADATA_PATH,
    TRANSFORMER_METRICS_PATH,
    load_json,
)

st.title("Model comparison")
metrics = load_json(str(METRICS_PATH))
transformer_metrics = load_json(str(TRANSFORMER_METRICS_PATH))
if not metrics and not transformer_metrics:
    st.info(
        "No measured model results are available. Train the baseline or an authorized "
        "transformer run first."
    )
    st.stop()

records = []
if metrics:
    records.append(
        (
            "FEVER TF-IDF + Logistic Regression",
            metrics["dataset"],
            metrics["task"],
            metrics["heldout"],
        )
    )
if transformer_metrics:
    records.append(
        (
            "AfrIFact XLM-R",
            transformer_metrics["dataset"],
            transformer_metrics["task"],
            transformer_metrics["heldout"],
        )
    )

st.markdown("### Measured held-out runs")
summary_rows = []
for name, dataset, task, heldout in records:
    summary_rows.append(
        {
            "Model": name,
            "Dataset": dataset,
            "Task": task,
            "Examples": heldout["n_examples"],
            "Accuracy": heldout["accuracy"],
            "Macro F1": heldout["macro_f1"],
            "Weighted F1": heldout["weighted_f1"],
        }
    )
st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

baseline_meta = load_json(str(RUN_METADATA_PATH)) if metrics else None
transformer_meta = load_json(str(TRANSFORMER_METADATA_PATH)) if transformer_metrics else None
comparable = bool(
    baseline_meta
    and transformer_meta
    and baseline_meta.get("heldout_sample_ids_sha256")
    == transformer_meta.get("heldout_sample_ids_sha256")
    and metrics.get("heldout", {}).get("labels")
    == transformer_metrics.get("heldout", {}).get("labels")
    and metrics.get("task") == transformer_metrics.get("task")
)
if len(records) > 1 and not comparable:
    st.warning(
        "These runs use different data, tasks, or held-out examples. They are shown separately; "
        "a direct score comparison would be invalid."
    )
elif comparable:
    st.success(
        "These runs share the same held-out example IDs, label inventory, and task definition."
    )

for name, dataset, task, heldout in records:
    st.markdown(f"### {name} · {dataset}")
    st.caption(f"Task: {task} · n={heldout['n_examples']:,}")
    if metrics and name.startswith("FEVER"):
        candidate_rows = [
            {
                "Validation candidate": candidate["name"],
                "Validation macro F1": candidate["validation"]["macro_f1"],
                "Validation accuracy": candidate["validation"]["accuracy"],
                "Selected": candidate["name"] == metrics["selected_candidate"],
            }
            for candidate in metrics.get("validation_candidates", [])
        ]
        st.dataframe(pd.DataFrame(candidate_rows), hide_index=True, use_container_width=True)
    metric_rows = pd.DataFrame(
        [
            {
                "Accuracy": heldout["accuracy"],
                "Macro precision": heldout["macro_precision"],
                "Macro recall": heldout["macro_recall"],
                "Macro F1": heldout["macro_f1"],
                "Weighted F1": heldout["weighted_f1"],
            }
        ]
    ).melt(var_name="Metric", value_name="Score")
    st.plotly_chart(
        px.bar(metric_rows, x="Metric", y="Score", range_y=[0, 1]), use_container_width=True
    )
    per_class = pd.DataFrame.from_dict(heldout["per_class"], orient="index").rename_axis("Label")
    st.dataframe(per_class, use_container_width=True)
    labels = heldout["labels"]
    confusion = heldout["confusion_matrix"]
    st.plotly_chart(
        go.Figure(
            data=go.Heatmap(
                z=confusion,
                x=[x.replace("_", " ") for x in labels],
                y=[x.replace("_", " ") for x in labels],
                text=[[f"{value:,}" for value in row] for row in confusion],
                texttemplate="%{text}",
                colorscale="Blues",
                colorbar={"title": "Claims"},
                hovertemplate="True %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
            )
        ).update_layout(xaxis_title="Predicted label", yaxis_title="True label"),
        use_container_width=True,
    )
if not transformer_metrics:
    st.info(
        "Transformer comparison is pending a real fine-tuning and evaluation run on an identical "
        "dataset split."
    )

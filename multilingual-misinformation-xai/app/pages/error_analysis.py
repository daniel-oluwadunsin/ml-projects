from __future__ import annotations

import streamlit as st

from app.ui import DEV_PATH, PREDICTIONS_PATH, load_prediction_rows
from ml.data.loaders import load_fever_jsonl

st.title("Error analysis")
predictions = load_prediction_rows(str(PREDICTIONS_PATH))
if not predictions:
    st.info("Held-out prediction rows are stored locally after baseline evaluation; none are available yet.")
    st.stop()

errors = [row for row in predictions if row["true_label"] != row["predicted_label"]]
st.metric("Held-out errors", f"{len(errors):,}", help="Count from the measured FEVER labelled development predictions.")
st.caption("Error categories have not been manually annotated; no category is inferred automatically.")
if not DEV_PATH.is_file():
    st.info("Local FEVER development text is missing. Prediction IDs are shown without reproducing raw claims.")
    st.dataframe(errors, hide_index=True, use_container_width=True)
else:
    records = load_fever_jsonl(DEV_PATH, "official_labelled_dev")
    claim_by_id = {record.sample_id: record.claim for record in records}
    rows = [{
        **row,
        "claim": claim_by_id.get(row["sample_id"], "[claim not found locally]"),
    } for row in errors]
    if not rows:
        st.success("No incorrect predictions in the available held-out output.")
    else:
        selected = st.selectbox("Filter by reference label", ["All"] + sorted({row["true_label"] for row in rows}))
        if selected != "All":
            rows = [row for row in rows if row["true_label"] == selected]
        st.dataframe(rows, hide_index=True, use_container_width=True)

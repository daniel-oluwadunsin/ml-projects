from __future__ import annotations

import json

import streamlit as st

from app.ui import DEV_PATH, PREDICTIONS_PATH, load_prediction_rows

st.title("Error analysis")
predictions = load_prediction_rows(str(PREDICTIONS_PATH))
if not predictions:
    st.info(
        "Held-out prediction rows are stored locally after baseline evaluation; none are "
        "available yet."
    )
    st.stop()

errors = [row for row in predictions if row["true_label"] != row["predicted_label"]]
st.metric(
    "Held-out errors",
    f"{len(errors):,}",
    help="Count from the measured FEVER labelled development predictions.",
)
st.caption(
    "Error categories have not been manually annotated; no category is inferred automatically."
)
if not DEV_PATH.is_file():
    st.info(
        "Local FEVER development text is missing. Prediction IDs are shown without reproducing "
        "raw claims."
    )
    st.dataframe(errors, hide_index=True, use_container_width=True)
else:
    if not errors:
        st.success("No incorrect predictions in the available held-out output.")
    else:
        selected = st.selectbox(
            "Filter by reference label", ["All"] + sorted({row["true_label"] for row in errors})
        )
        filtered = [row for row in errors if selected == "All" or row["true_label"] == selected]
        page_size = st.selectbox("Maximum examples to display", [25, 50, 100], index=1)
        rows = filtered[:page_size]
        wanted_ids = {row["sample_id"] for row in rows}
        claim_by_id = {}
        with DEV_PATH.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                sample_id = str(record.get("id"))
                if sample_id in wanted_ids and isinstance(record.get("claim"), str):
                    claim_by_id[sample_id] = record["claim"]
                    wanted_ids.remove(sample_id)
                if not wanted_ids:
                    break
        rows = [
            {**row, "claim": claim_by_id.get(row["sample_id"], "[claim not found locally]")}
            for row in rows
        ]
        st.caption(f"Showing {len(rows)} of {len(filtered):,} selected errors.")
        st.dataframe(rows, hide_index=True, use_container_width=True)

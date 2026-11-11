from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui import DATA_SUMMARY_PATH, load_json

st.title("Dataset explorer")
st.markdown("### FEVER · local English baseline")
st.markdown(
    "Source: [official FEVER site](https://fever.ai/dataset/fever.html). Raw files remain local."
)
summary = load_json(str(DATA_SUMMARY_PATH))
if not summary:
    st.info("Run `uv run python scripts/prepare_data.py --dataset fever` to inspect local files.")
else:
    st.caption(summary["note"])
    for split, values in summary.items():
        if not isinstance(values, dict) or "n_examples" not in values:
            continue
        st.markdown(f"#### {split.replace('_', ' ').title()} · {values['n_examples']:,} examples")
        label_frame = pd.DataFrame(values["label_counts"].items(), columns=["Label", "Examples"])
        language_frame = pd.DataFrame(
            values["language_counts"].items(), columns=["Language", "Examples"]
        )
        st.plotly_chart(px.bar(label_frame, x="Label", y="Examples"), use_container_width=True)
        st.dataframe(language_frame, hide_index=True, use_container_width=True)
        st.json(
            {
                "duplicate claim groups": values["duplicate_claim_groups"],
                "evidence annotations present": values["evidence_annotation_present"],
                "textual evidence passages available in this file": values[
                    "evidence_text_available"
                ],
                "claim character lengths": values["claim_character_length"],
            }
        )

st.markdown("### Target and secondary sources")
st.write(
    "AfrIFact: access gated; inspect the actual schema after approval. X-Fact: reviewed, but its "
    "documented language inventory does not include the four target languages."
)
st.caption("The app does not display gated samples or redistribute dataset rows.")

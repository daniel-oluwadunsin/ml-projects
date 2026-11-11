from __future__ import annotations

import streamlit as st

from app.ui import read_docs

st.title("Research methodology")
st.markdown(read_docs("docs/methodology.md"))
st.markdown("### Dataset access and schema")
st.markdown(read_docs("docs/dataset-investigation.md"))

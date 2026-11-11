"""Multipage Streamlit research dashboard."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Multilingual Claim Classification Research",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/overview.py", title="Research overview", icon="📘", default=True),
    st.Page("pages/claim_analysis.py", title="Claim analysis", icon="🔍"),
    st.Page("pages/model_comparison.py", title="Model comparison", icon="📊"),
    st.Page("pages/language_performance.py", title="Language performance", icon="🌍"),
    st.Page("pages/explainability.py", title="Explainability", icon="🧭"),
    st.Page("pages/error_analysis.py", title="Error analysis", icon="⚠️"),
    st.Page("pages/dataset_explorer.py", title="Dataset explorer", icon="🗂️"),
    st.Page("pages/methodology.py", title="Methodology", icon="🧪"),
    st.Page("pages/about.py", title="About and references", icon="📚"),
]
navigation = st.navigation(pages)
navigation.run()

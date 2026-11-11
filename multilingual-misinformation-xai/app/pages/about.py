from __future__ import annotations

import streamlit as st

from app.ui import read_docs

st.title("About and references")
st.write(
    "This is an independent research portfolio project. It is not endorsed by Erasmus Mundus, "
    "EMILDAI, EMLDS, or a university."
)
st.markdown("### Research references")
st.markdown(
    "- [AfrIFact dataset card](https://huggingface.co/datasets/masakhane/AfrIFact) · "
    "[EMNLP paper](https://aclanthology.org/2026.emnlp-main.133/)\n"
    "- [X-Fact dataset card](https://huggingface.co/datasets/utahnlp/x-fact) · "
    "[ACL paper](https://aclanthology.org/2021.acl-short.86/)\n"
    "- [FEVER dataset and terms](https://fever.ai/dataset/fever.html) · "
    "[NAACL-HLT paper](https://aclanthology.org/N18-1074/)\n"
    "- [XLM-RoBERTa Base model card](https://huggingface.co/FacebookAI/xlm-roberta-base) · "
    "[ACL paper](https://aclanthology.org/2020.acl-main.747/)"
)
st.markdown("### Ethical considerations")
st.markdown(read_docs("docs/limitations.md"))
st.markdown("### Code license")
st.write("Project code is MIT licensed. Dataset licenses, access conditions, and pretrained model terms remain separate.")

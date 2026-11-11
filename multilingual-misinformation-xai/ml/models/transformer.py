"""Optional XLM-R sequence-classification helpers."""

from __future__ import annotations

from typing import Any


def available_device(torch_module: Any | None = None) -> str:
    """Choose CUDA, then Apple MPS, then CPU without requiring optional torch at import."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError(
                "Transformer training requires `uv sync --extra transformer`"
            ) from exc
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_finetuned_transformer(model_dir: str):
    """Load a task-fine-tuned model; base XLM-R is intentionally not substituted."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install transformer dependencies with `uv sync --extra transformer`"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    return tokenizer, model

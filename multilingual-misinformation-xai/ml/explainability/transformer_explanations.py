"""Optional Integrated Gradients attribution for a fine-tuned transformer."""

from __future__ import annotations

from typing import Any


def integrated_gradients_tokens(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    target_class: int,
    max_length: int = 256,
    n_steps: int = 32,
) -> list[dict[str, float | str]]:
    """Attribute a selected class logit to subword embeddings using Captum IG.

    Attribution is relative to a zero-embedding baseline and summarizes each
    token by summing the signed attribution across its embedding dimensions.
    It is not evidence that the claim is true.
    """
    try:
        import torch
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise RuntimeError(
            "Install optional explanation dependencies with `uv sync --extra explainability`"
        ) from exc
    if not text.strip():
        raise ValueError("text must not be empty")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length, padding=False
    ).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids).detach()
    baseline = torch.zeros_like(embeddings)

    def score(inputs_embeds):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits[:, target_class]

    attributor = IntegratedGradients(score)
    attribution = attributor.attribute(
        embeddings,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=max(2, n_steps),
    ).sum(dim=-1)[0]
    if attention_mask is not None:
        attribution = attribution * attention_mask[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return [
        {"token": str(token), "attribution": float(value)}
        for token, value in zip(tokens, attribution.detach().cpu().tolist(), strict=True)
    ]

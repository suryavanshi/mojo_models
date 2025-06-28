"""Weight conversion for Gemma3n."""
from __future__ import annotations

from max.graph.weights import WeightData, Weights

# Maps from Safetensor to MAX weight names for Gemma3n.
GEMMA3N_SAFETENSOR_MAP: dict[str, str] = {
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "lm_head.": "language_model.lm_head.",
    "model.layers.": "language_model.layers.",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        max_name = weight_name
        for before, after in GEMMA3N_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    return new_state_dict

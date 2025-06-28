from __future__ import annotations

from typing import Dict

from max.graph.weights import WeightData, Weights


def convert_safetensor_state_dict(state_dict: Dict[str, Weights], **unused_kwargs) -> Dict[str, WeightData]:
    """Return state dict in MAX format without modifications."""
    return {k: v.data() for k, v in state_dict.items()}

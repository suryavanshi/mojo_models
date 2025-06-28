"""SupportedArchitecture registration for Gemma3n."""
from max.graph.weights import WeightsFormat
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import PipelineTask
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import Gemma3nModel

# Example architecture registration. The example_repo_ids correspond to HF
# checkpoints available for Gemma3n.

gemma3n_arch = SupportedArchitecture(
    name="Gemma3nForCausalLM",
    example_repo_ids=["google/gemma-3n-E2B-it"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED]},
    pipeline_model=Gemma3nModel,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
)

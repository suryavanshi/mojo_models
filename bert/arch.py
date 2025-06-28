from max.graph.weights import WeightsFormat
from max.pipelines.core import PipelineTask
from max.pipelines.lib import SupportedArchitecture, SupportedEncoding, TextTokenizer

from .bert import BertPipelineModel

bert_arch = SupportedArchitecture(
    name="BertModel",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=["bert-base-uncased"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [],
        SupportedEncoding.bfloat16: [],
    },
    pipeline_model=BertPipelineModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
)

"""Pipeline model implementation for Gemma3n."""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnLogits
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from transformers import AutoConfig

from gemma3n.gemma3n import Gemma3n
from gemma3n.model_config import Gemma3nConfig

logger = logging.getLogger("max.pipelines")


class Gemma3nInputs(ModelInputs):
    """Input container for Gemma3n model."""

    tokens: np.ndarray | Tensor
    input_row_offsets: np.ndarray | Tensor

    def __init__(
        self,
        tokens: np.ndarray | Tensor,
        input_row_offsets: np.ndarray | Tensor,
        return_n_logits: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.kv_cache_inputs = kv_cache_inputs
        self.return_n_logits = return_n_logits


class Gemma3nModel(PipelineModel[TextContext], KVCacheMixin):
    """Gemma3n pipeline model for text generation."""

    model: Model

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self.model = self.load_model(session)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len
        return huggingface_config.max_position_embeddings

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Gemma3nConfig.get_kv_params(
            huggingface_config, n_devices, kv_cache_config, cache_dtype
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return Gemma3nConfig.get_num_layers(huggingface_config)

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        return estimate_kv_cache_size(
            params=Gemma3nConfig.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=Gemma3nConfig.get_num_layers(
                huggingface_config=huggingface_config
            ),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def load_model(self, session: InferenceSession) -> Model:
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        logger.info("Building and compiling model...")
        before = time.perf_counter()
        graph = self._build_graph()
        model = session.load(graph, weights_registry=self.state_dict)
        after = time.perf_counter()
        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )
        return model

    _strict_state_dict_loading = True

    def _build_graph(self):
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        huggingface_config = self.huggingface_config
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        model_config = Gemma3nConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            logits_postprocessor=None,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )
        nn_model = Gemma3n(model_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        self.state_dict = nn_model.state_dict(auto_initialize=False)

        with Graph(
            getattr(self.huggingface_config, "model_type", "Gemma3n"),
            input_types=[
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *self.kv_manager.input_symbols()[0],
            ],
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            outputs = nn_model(
                tokens.tensor,
                input_row_offsets.tensor,
                [inp.tensor for inp in kv_cache_inputs],
                return_n_logits=return_n_logits.tensor,
            )
            graph.output(*outputs)
        return graph

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(Gemma3nInputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *curr_kv_cache_inputs,
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Tensor, model_outputs[1]),
                next_token_logits=cast(Tensor, model_outputs[0]),
                logit_offsets=cast(Tensor, model_outputs[2]),
            )
        else:
            return ModelOutputs(
                logits=cast(Tensor, model_outputs[0]),
                next_token_logits=cast(Tensor, model_outputs[0]),
            )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        assert kv_cache_inputs is not None
        kv_cache_inputs = cast(KVCacheInputsSequence, kv_cache_inputs)
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch], dtype=np.uint32
        )
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])

        return Gemma3nInputs(
            tokens=Tensor.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Tensor.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Tensor, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        prev_model_inputs = cast(Gemma3nInputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        return Gemma3nInputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    def load_kv_manager(
        self, session: InferenceSession, available_cache_memory: int | None
    ) -> KVCacheManager:
        return load_kv_manager(
            params=Gemma3nConfig.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=Gemma3nConfig.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )

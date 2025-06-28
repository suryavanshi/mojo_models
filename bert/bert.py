from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Optional, cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import Weights, WeightsAdapter
from max.nn import EmbeddingV1, LayerNormV1, LinearV1, Sequential, ReturnLogits
from max.nn.layer import Layer
from max.pipelines.core import TextContext
from max.pipelines.dataprocessing import collate_batch
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    upper_bounded_default,
)
from transformers import AutoConfig

from .model_config import BertConfig

logger = logging.getLogger("max.pipelines")

PAD_VALUE = 0


def _quantization_encoding(pipeline_config: PipelineConfig) -> QuantizationEncoding | None:
    if supported_encoding := pipeline_config.model_config.quantization_encoding:
        return supported_encoding.quantization_encoding
    return None


class BertEmbeddings(Layer):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        config = huggingface_config
        self.word_embeddings = EmbeddingV1(
            weights.word_embeddings.weight.allocate(
                DType.float32,
                [config.vocab_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            device,
        )
        self.position_embeddings = EmbeddingV1(
            weights.position_embeddings.weight.allocate(
                DType.float32,
                [config.max_position_embeddings, config.hidden_size],
            ).cast(dtype),
            device,
        )
        self.token_type_embeddings = EmbeddingV1(
            weights.token_type_embeddings.weight.allocate(
                DType.float32,
                [config.type_vocab_size, config.hidden_size],
            ).cast(dtype),
            device,
        )
        self.layer_norm = LayerNormV1(
            weight=weights.LayerNorm.weight.allocate(
                DType.float32, [config.hidden_size]
            ),
            bias=weights.LayerNorm.bias.allocate(
                DType.float32, [config.hidden_size]
            ),
            eps=config.layer_norm_eps,
        )
        self.position_ids = weights.position_ids.allocate(
            DType.int64,
            [1, config.max_position_embeddings],
        )

    def __call__(self, input_ids: TensorValue, token_type_ids: TensorValue) -> TensorValue:
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(position_ids)
        token_types = self.token_type_embeddings(token_type_ids)
        embeddings = words + positions + token_types
        return self.layer_norm(embeddings)


class BertSelfAttention(Layer):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
    ) -> None:
        config = huggingface_config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q = LinearV1(
            weights.q.weight.allocate(
                DType.float32,
                [self.all_head_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.q.bias.allocate(
                DType.float32,
                [self.all_head_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )
        self.k = LinearV1(
            weights.k.weight.allocate(
                DType.float32,
                [self.all_head_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.k.bias.allocate(
                DType.float32,
                [self.all_head_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )
        self.v = LinearV1(
            weights.v.weight.allocate(
                DType.float32,
                [self.all_head_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.v.bias.allocate(
                DType.float32,
                [self.all_head_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )
        self.o = LinearV1(
            weights.o.weight.allocate(
                DType.float32,
                [config.hidden_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.o.bias.allocate(
                DType.float32,
                [config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )

    def transpose_for_scores(self, x: TensorValue) -> TensorValue:
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = ops.reshape(x, new_x_shape)
        return ops.permute(x, [0, 2, 1, 3])

    def __call__(self, hidden_states, attention_mask: TensorValue) -> TensorValue:
        q = self.transpose_for_scores(self.q(hidden_states))
        k = self.transpose_for_scores(self.k(hidden_states))
        v = self.transpose_for_scores(self.v(hidden_states))
        attention_scores = q @ k.transpose(-1, -2)
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = ops.softmax(attention_scores)
        context = attention_probs @ v
        context = ops.permute(context, [0, 2, 1, 3])
        new_c_shape = context.shape[:-2] + [self.all_head_size]
        context = ops.reshape(context, new_c_shape)
        return self.o(context)


class BertAttention(Layer):
    def __init__(self, pipeline_config: PipelineConfig, weights: Weights, huggingface_config: AutoConfig, dtype: DType) -> None:
        config = huggingface_config
        self.attn = BertSelfAttention(pipeline_config, weights.attn, huggingface_config, dtype)
        self.layer_norm = LayerNormV1(
            weight=weights.LayerNorm.weight.allocate(DType.float32, [config.hidden_size]),
            bias=weights.LayerNorm.bias.allocate(DType.float32, [config.hidden_size]),
            eps=config.layer_norm_eps,
        )

    def __call__(self, hidden_states: TensorValue, attention_mask: TensorValue) -> TensorValue:
        attn_output = self.attn(hidden_states, attention_mask)
        return self.layer_norm(attn_output + hidden_states)


_ACTIVATIONS = {
    "gelu": ops.gelu,
    "relu": ops.relu,
    "silu": ops.silu,
    "sigmoid": ops.sigmoid,
    "tanh": ops.tanh,
}


class BertIntermediate(Layer):
    def __init__(self, pipeline_config: PipelineConfig, weights: Weights, huggingface_config: AutoConfig, dtype: DType) -> None:
        config = huggingface_config
        self.dense = LinearV1(
            weights.dense.weight.allocate(
                DType.float32,
                [config.intermediate_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.dense.bias.allocate(
                DType.float32,
                [config.intermediate_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )
        self.intermediate_act_fn = _ACTIVATIONS[config.hidden_act]

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(Layer):
    def __init__(self, pipeline_config: PipelineConfig, weights: Weights, huggingface_config: AutoConfig, dtype: DType) -> None:
        config = huggingface_config
        self.dense = LinearV1(
            weights.dense.weight.allocate(
                DType.float32,
                [config.hidden_size, config.intermediate_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.dense.bias.allocate(
                DType.float32,
                [config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )
        self.layer_norm = LayerNormV1(
            weight=weights.LayerNorm.weight.allocate(DType.float32, [config.hidden_size]),
            bias=weights.LayerNorm.bias.allocate(DType.float32, [config.hidden_size]),
            eps=config.layer_norm_eps,
        )

    def __call__(self, hidden_states: TensorValue, input_tensor: TensorValue) -> TensorValue:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(Layer):
    def __init__(self, pipeline_config: PipelineConfig, weights: Weights, huggingface_config: AutoConfig, dtype: DType) -> None:
        self.attention = BertAttention(pipeline_config, weights.attention, huggingface_config, dtype)
        self.intermediate = BertIntermediate(pipeline_config, weights.intermediate, huggingface_config, dtype)
        self.output = BertOutput(pipeline_config, weights.output, huggingface_config, dtype)

    def __call__(self, hidden_states: TensorValue, attention_mask: TensorValue) -> TensorValue:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(Layer):
    def __init__(self, pipeline_config: PipelineConfig, weights: Weights, huggingface_config: AutoConfig, dtype: DType, device: DeviceRef) -> None:
        config = huggingface_config
        num_hidden_layers = config.num_hidden_layers
        self.layer = Sequential([
            BertLayer(pipeline_config, weights.layer[n], huggingface_config, dtype)
            for n in range(num_hidden_layers)
        ])
        self.num_attention_heads = config.num_attention_heads

    def __call__(self, hidden_states: TensorValue, attention_mask: TensorValue) -> TensorValue:
        for layer in self.layer.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertModel(Layer):
    def __init__(self, pipeline_config: PipelineConfig, weights: Weights, huggingface_config: AutoConfig, dtype: DType, device: DeviceRef) -> None:
        self.embeddings = BertEmbeddings(pipeline_config, weights.embeddings, huggingface_config, dtype, device)
        self.encoder = BertEncoder(pipeline_config, weights.encoder, huggingface_config, dtype, device)
        self.pool_outputs = pipeline_config.pool_embeddings

    def __call__(self, input_ids: TensorValue, token_type_ids: TensorValue, attention_mask: TensorValue) -> TensorValue:
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        extended_attention_mask = ops.reshape(attention_mask, ("batch_size", 1, 1, "seq_len"))
        extended_attention_mask = (1 - extended_attention_mask) * ops.constant(
            np.finfo(np.float32).min, DType.float32, device=attention_mask.device
        )
        encoded_results = self.encoder(embedding_output, attention_mask=extended_attention_mask)
        if self.pool_outputs:
            encoded_results = encoded_results.transpose(1, 2)
            input_mask_expanded = ops.broadcast_to(
                ops.unsqueeze(attention_mask, 1),
                ("batch_size", encoded_results.shape[1], "seq_len"),
            )
            input_lengths = ops.max(
                ops.sum(input_mask_expanded),
                ops.constant(1e-9, DType.float32, device=input_mask_expanded.device),
            )
            pooled_output = ops.sum(encoded_results * input_mask_expanded) / input_lengths
            return ops.squeeze(pooled_output, 2)
        else:
            return encoded_results


def build_graph(
    pipeline_config: PipelineConfig,
    weights: Weights,
    huggingface_config: AutoConfig,
    dtype: DType,
    input_device: DeviceRef,
) -> Graph:
    input_ids_type = TensorType(DType.int64, shape=["batch_size", "seq_len"], device=input_device)
    token_type_type = TensorType(DType.int64, shape=["batch_size", "seq_len"], device=input_device)
    attention_mask_type = TensorType(DType.float32, shape=["batch_size", "seq_len"], device=input_device)

    with Graph("bert", input_types=[input_ids_type, token_type_type, attention_mask_type]) as graph:
        bert = BertModel(pipeline_config, weights, huggingface_config, dtype, device=input_device)
        input_ids = graph.inputs[0].tensor
        token_type_ids = graph.inputs[1].tensor
        attention_mask = graph.inputs[2].tensor
        graph.output(bert(input_ids, token_type_ids, attention_mask))
    return graph


class BertInputs(ModelInputs):
    next_tokens_batch: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor

    def __init__(self, next_tokens_batch: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> None:
        self.next_tokens_batch = next_tokens_batch
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.kv_cache_inputs = None


class BertPipelineModel(PipelineModel[TextContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
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

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return BertConfig.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return BertConfig.get_num_layers(huggingface_config)

    @classmethod
    def calculate_max_seq_len(cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for Bert, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(BertInputs, model_inputs)
        model_outputs = self.model.execute(
            model_inputs.next_tokens_batch,
            model_inputs.token_type_ids,
            model_inputs.attention_mask,
        )
        return ModelOutputs(logits=cast(Tensor, model_outputs[0]))

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> BertInputs:
        tokens = [ctx.next_tokens for ctx in context_batch]
        pad_value = getattr(self.huggingface_config, "pad_token_id", PAD_VALUE)
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=pad_value,
            batch_size=len(tokens),
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )
        attention_mask = (next_tokens_batch != pad_value).astype(np.float32)
        token_type_ids = np.zeros_like(next_tokens_batch, dtype=np.int64)
        return BertInputs(
            next_tokens_batch=Tensor.from_numpy(next_tokens_batch).to(self.devices[0]),
            attention_mask=Tensor.from_numpy(attention_mask).to(self.devices[0]),
            token_type_ids=Tensor.from_numpy(token_type_ids).to(self.devices[0]),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> BertInputs:
        raise NotImplementedError("Bert does not support preparing next tokens inputs.")

    def load_model(self, session: InferenceSession) -> Model:
        logger.info("Building and compiling model...")
        before = time.perf_counter()
        graph = build_graph(
            self.pipeline_config,
            self.weights,
            self.huggingface_config,
            self.dtype,
            DeviceRef.from_device(self.devices[0]),
        )
        model = session.load(graph, weights_registry=self.weights.allocated_weights)
        after = time.perf_counter()
        logger.info(f"Building and compiling model took {after - before:.6f} seconds")
        return model

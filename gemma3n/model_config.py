"""Gemma3n configuration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn import LinearScalingParams, ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MAXModelConfigBase,
    PipelineConfig,
    RopeType,
)
from transformers import AutoConfig


@dataclass
class Gemma3nConfigBase(MAXModelConfigBase):
    """Base configuration for Gemma3n models."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_activation: str
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    rope_theta: float
    attention_bias: bool
    query_pre_attn_scalar: float | None
    sliding_window: int
    final_logit_softcapping: float | None
    attn_logit_softcapping: int | None
    rope_scaling: LinearScalingParams | None
    rope_local_base_freq: float
    sliding_window_pattern: int
    dtype: DType
    devices: list[DeviceRef]
    interleaved_rope_weights: bool
    return_logits: ReturnLogits
    kv_params: KVCacheParams


@dataclass
class Gemma3nConfig(MAXModelConfig, Gemma3nConfigBase):
    """Full MAX configuration for Gemma3n."""

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len
        return huggingface_config.max_position_embeddings

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        dtype: DType,
        n_devices: int,
        logits_postprocessor: Callable[[TensorValue], TensorValue] | None,
        cache_dtype: DType,
        kv_cache_config: KVCacheConfig,
        return_logits: ReturnLogits,
        norm_method: Literal["rms_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> "Gemma3nConfig":
        _weights_format = weights_format(pipeline_config.model_config.weight_path)
        interleaved_rope_weights = (
            _weights_format == WeightsFormat.gguf
            and pipeline_config.model_config.rope_type == RopeType.normal
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model_config.device_specs
        ]

        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "language_model.lm_head.weight" not in state_dict
        )

        rope_scaling_params = None
        rope_scaling = huggingface_config.rope_scaling
        if rope_scaling is not None:
            rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
            if rope_type == "linear":
                rope_scaling_params = LinearScalingParams(factor=rope_scaling["factor"])

        hidden_activation = huggingface_config.hidden_activation

        return Gemma3nConfig(
            vocab_size=huggingface_config.vocab_size,
            hidden_size=huggingface_config.hidden_size,
            intermediate_size=huggingface_config.intermediate_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            hidden_activation=hidden_activation,
            max_position_embeddings=huggingface_config.max_position_embeddings,
            rms_norm_eps=huggingface_config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=huggingface_config.rope_theta,
            attention_bias=huggingface_config.attention_bias,
            query_pre_attn_scalar=huggingface_config.query_pre_attn_scalar,
            sliding_window=huggingface_config.sliding_window,
            final_logit_softcapping=huggingface_config.final_logit_softcapping,
            attn_logit_softcapping=huggingface_config.attn_logit_softcapping,
            rope_scaling=rope_scaling_params,
            rope_local_base_freq=huggingface_config.rope_local_base_freq,
            sliding_window_pattern=huggingface_config.sliding_window_pattern,
            dtype=dtype,
            devices=device_refs,
            interleaved_rope_weights=interleaved_rope_weights,
            return_logits=return_logits,
            kv_params=Gemma3nConfig.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=n_devices,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
        )

"""KV cache size estimation for LLM-Lander."""

from __future__ import annotations

from llm_lander.model_profiler import ModelConfig, ModelProfiler, QuantizationType


class KVCacheCalculator:
    """Compute KV cache memory usage."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.config = model_config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config["num_layers"] <= 0:
            raise ValueError("num_layers must be positive")
        if self.config["hidden_size"] <= 0:
            raise ValueError("hidden_size must be positive")
        if self.config["num_attention_heads"] <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.config["num_key_value_heads"] <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.config["num_key_value_heads"] > self.config["num_attention_heads"]:
            raise ValueError("num_key_value_heads must be <= num_attention_heads")

    def calculate_kv_cache_size(
        self,
        context_length: int,
        quantization: QuantizationType = "FP16",
    ) -> float:
        """Calculate KV cache size in MB."""
        if context_length < 0:
            raise ValueError("context_length must be non-negative")

        num_layers = self.config["num_layers"]
        kv_heads = self.config["num_key_value_heads"]
        hidden_size = self.config["hidden_size"]
        num_attention_heads = self.config["num_attention_heads"]

        head_dim = hidden_size // num_attention_heads
        bytes_per_param = ModelProfiler.BYTES_PER_PARAM[quantization]

        kv_cache_bytes = 2 * num_layers * kv_heads * head_dim * context_length * bytes_per_param
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)
        return kv_cache_mb

    def calculate_kv_cache_per_token(self, quantization: QuantizationType = "FP16") -> float:
        """Calculate KV cache size per token in MB."""
        return self.calculate_kv_cache_size(context_length=1, quantization=quantization)

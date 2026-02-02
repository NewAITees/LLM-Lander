"""Model profiling utilities for LLM-Lander."""

from __future__ import annotations

from typing import Literal, TypedDict

QuantizationType = Literal["FP32", "FP16", "INT8", "INT4"]


class ModelConfig(TypedDict):
    """Model configuration for memory estimation."""

    num_params: float  # in billions (B)
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int


class ModelProfiler:
    """Compute model weight and activation memory estimates."""

    BYTES_PER_PARAM: dict[QuantizationType, float] = {
        "FP32": 4.0,
        "FP16": 2.0,
        "INT8": 1.0,
        "INT4": 0.5,
    }

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config["num_params"] <= 0:
            raise ValueError("num_params must be positive")
        if self.config["num_layers"] <= 0:
            raise ValueError("num_layers must be positive")
        if self.config["hidden_size"] <= 0:
            raise ValueError("hidden_size must be positive")
        if self.config["num_attention_heads"] <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.config["num_key_value_heads"] <= 0:
            raise ValueError("num_key_value_heads must be positive")

    def calculate_weight_size(self, quantization: QuantizationType = "FP16") -> float:
        """Calculate model weight size in MB."""
        num_params_total = self.config["num_params"] * 1e9
        bytes_per_param = self.BYTES_PER_PARAM[quantization]
        weight_size_mb = (num_params_total * bytes_per_param) / (1024 * 1024)
        return weight_size_mb

    def estimate_activation_memory(self, batch_size: int = 1) -> float:
        """Estimate activation memory in MB."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]
        activation_mb = (batch_size * hidden_size * num_layers * 4) / (1024 * 1024)
        return activation_mb

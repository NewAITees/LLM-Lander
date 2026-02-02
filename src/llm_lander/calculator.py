"""VRAM calculation and reverse context estimation."""

from __future__ import annotations

from typing import Literal, TypedDict

from llm_lander.kv_cache import KVCacheCalculator
from llm_lander.model_profiler import ModelProfiler, QuantizationType

BackendType = Literal["llama.cpp", "vLLM", "AutoGPTQ", "Transformers"]


class CalculationResult(TypedDict):
    """Result of a VRAM calculation."""

    max_context_length: int
    model_weight_mb: float
    kv_cache_mb: float
    total_memory_mb: float
    fits_in_vram: bool


class VRAMCalculator:
    """Compute VRAM totals and search max context length."""

    BACKEND_OVERHEAD: dict[BackendType, float] = {
        "llama.cpp": 1.05,
        "vLLM": 1.10,
        "AutoGPTQ": 1.08,
        "Transformers": 1.15,
    }

    def __init__(
        self,
        model_profiler: ModelProfiler,
        kv_cache_calculator: KVCacheCalculator,
        safety_margin_mb: float = 500.0,
    ) -> None:
        if safety_margin_mb < 0:
            raise ValueError("safety_margin_mb must be non-negative")
        self.model_profiler = model_profiler
        self.kv_cache_calculator = kv_cache_calculator
        self.safety_margin_mb = safety_margin_mb

    def calculate_total_memory(
        self,
        context_length: int,
        quantization: QuantizationType = "FP16",
        backend: BackendType = "llama.cpp",
    ) -> float:
        """Calculate total memory usage in MB."""
        weight_mb = self.model_profiler.calculate_weight_size(quantization)
        kv_cache_mb = self.kv_cache_calculator.calculate_kv_cache_size(context_length, quantization)
        activation_mb = self.model_profiler.estimate_activation_memory()

        base_memory_mb = weight_mb + kv_cache_mb + activation_mb + self.safety_margin_mb
        total_memory_mb = base_memory_mb * self.BACKEND_OVERHEAD[backend]
        return total_memory_mb

    def find_max_context_length(
        self,
        free_vram_mb: float,
        quantization: QuantizationType = "FP16",
        backend: BackendType = "llama.cpp",
    ) -> int:
        """Binary search for the maximum context length that fits."""
        if free_vram_mb <= 0:
            raise ValueError("free_vram_mb must be positive")

        left, right = 1, 1_000_000
        max_context = 0

        while left <= right:
            mid = (left + right) // 2
            total_mb = self.calculate_total_memory(mid, quantization, backend)

            if total_mb <= free_vram_mb:
                max_context = mid
                left = mid + 1
            else:
                right = mid - 1

        return max_context

    def suggest_optimal_config(
        self, free_vram_mb: float, backend: BackendType = "llama.cpp"
    ) -> dict[str, CalculationResult]:
        """Return results for common quantization options."""
        if free_vram_mb <= 0:
            raise ValueError("free_vram_mb must be positive")

        results: dict[str, CalculationResult] = {}
        for quantization in ("FP16", "INT8", "INT4"):
            max_context = self.find_max_context_length(free_vram_mb, quantization, backend)
            weight_mb = self.model_profiler.calculate_weight_size(quantization)
            kv_cache_mb = self.kv_cache_calculator.calculate_kv_cache_size(
                max_context, quantization
            )
            total_mb = self.calculate_total_memory(max_context, quantization, backend)

            results[quantization] = {
                "max_context_length": max_context,
                "model_weight_mb": weight_mb,
                "kv_cache_mb": kv_cache_mb,
                "total_memory_mb": total_mb,
                "fits_in_vram": total_mb <= free_vram_mb,
            }

        return results

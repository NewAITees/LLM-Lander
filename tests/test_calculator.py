import pytest

from llm_lander.calculator import VRAMCalculator
from llm_lander.kv_cache import KVCacheCalculator
from llm_lander.model_profiler import ModelConfig, ModelProfiler


def _sample_config() -> ModelConfig:
    return {
        "num_params": 7.0,
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }


def test_calculate_total_memory_includes_overhead() -> None:
    config = _sample_config()
    profiler = ModelProfiler(config)
    kv_calc = KVCacheCalculator(config)
    calculator = VRAMCalculator(profiler, kv_calc, safety_margin_mb=500.0)

    total = calculator.calculate_total_memory(context_length=1024, quantization="FP16")

    weight = profiler.calculate_weight_size("FP16")
    kv_cache = kv_calc.calculate_kv_cache_size(1024, "FP16")
    activation = profiler.estimate_activation_memory()
    base = weight + kv_cache + activation + 500.0
    expected = base * calculator.BACKEND_OVERHEAD["llama.cpp"]
    assert total == pytest.approx(expected, rel=1e-6)


def test_find_max_context_length_binary_search() -> None:
    config = _sample_config()
    profiler = ModelProfiler(config)
    kv_calc = KVCacheCalculator(config)
    calculator = VRAMCalculator(profiler, kv_calc, safety_margin_mb=500.0)

    total_150 = calculator.calculate_total_memory(150, "FP16")
    total_151 = calculator.calculate_total_memory(151, "FP16")
    free_vram = (total_150 + total_151) / 2

    max_context = calculator.find_max_context_length(free_vram, "FP16")
    assert max_context == 150


def test_suggest_optimal_config_shape() -> None:
    config = _sample_config()
    profiler = ModelProfiler(config)
    kv_calc = KVCacheCalculator(config)
    calculator = VRAMCalculator(profiler, kv_calc, safety_margin_mb=500.0)

    results = calculator.suggest_optimal_config(24_000.0)

    for quantization in ("FP16", "INT8", "INT4"):
        assert quantization in results
        entry = results[quantization]
        assert entry["max_context_length"] >= 1
        assert entry["model_weight_mb"] > 0
        assert entry["kv_cache_mb"] >= 0
        assert entry["total_memory_mb"] > 0
        assert isinstance(entry["fits_in_vram"], bool)

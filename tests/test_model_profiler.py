import pytest

from llm_lander.model_profiler import ModelConfig, ModelProfiler


def _sample_config() -> ModelConfig:
    return {
        "num_params": 7.0,
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
    }


def test_calculate_weight_size_fp16() -> None:
    profiler = ModelProfiler(_sample_config())
    weight_mb = profiler.calculate_weight_size("FP16")
    expected = (7.0e9 * 2.0) / (1024 * 1024)
    assert weight_mb == pytest.approx(expected, rel=1e-6)


def test_calculate_weight_size_int4_ratio() -> None:
    profiler = ModelProfiler(_sample_config())
    weight_fp16 = profiler.calculate_weight_size("FP16")
    weight_int4 = profiler.calculate_weight_size("INT4")
    assert weight_int4 == pytest.approx(weight_fp16 * 0.25, rel=1e-6)


def test_estimate_activation_memory_default_batch() -> None:
    profiler = ModelProfiler(_sample_config())
    activation_mb = profiler.estimate_activation_memory()
    expected = (1 * 4096 * 32 * 4) / (1024 * 1024)
    assert activation_mb == pytest.approx(expected, rel=1e-6)

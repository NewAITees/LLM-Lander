import pytest

from llm_lander.kv_cache import KVCacheCalculator
from llm_lander.model_profiler import ModelConfig


def _sample_config() -> ModelConfig:
    return {
        "num_params": 7.0,
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }


def test_calculate_kv_cache_size_fp16() -> None:
    config = _sample_config()
    calculator = KVCacheCalculator(config)
    kv_mb = calculator.calculate_kv_cache_size(context_length=1024, quantization="FP16")

    head_dim = config["hidden_size"] // config["num_attention_heads"]
    expected_bytes = (
        2 * config["num_layers"] * config["num_key_value_heads"] * head_dim * 1024 * 2.0
    )
    expected_mb = expected_bytes / (1024 * 1024)
    assert kv_mb == pytest.approx(expected_mb, rel=1e-6)


def test_calculate_kv_cache_per_token() -> None:
    config = _sample_config()
    calculator = KVCacheCalculator(config)
    per_token = calculator.calculate_kv_cache_per_token("INT8")
    single = calculator.calculate_kv_cache_size(context_length=1, quantization="INT8")
    assert per_token == pytest.approx(single, rel=1e-6)

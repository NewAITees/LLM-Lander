"""Minimal CLI for LLM-Lander calculations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import cast

from llm_lander.calculator import CalculationResult, VRAMCalculator
from llm_lander.kv_cache import KVCacheCalculator
from llm_lander.model_profiler import ModelConfig, ModelProfiler, QuantizationType


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-Lander VRAM/Context calculator (MVP CLI)")
    parser.add_argument("--free-vram-mb", type=float, required=True)
    parser.add_argument("--num-params-b", type=float, default=7.0)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-key-value-heads", type=int, default=32)
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["FP16", "INT8", "INT4", "FP32"],
        default=None,
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["llama.cpp", "vLLM", "AutoGPTQ", "Transformers"],
        default="llama.cpp",
    )
    parser.add_argument("--safety-margin-mb", type=float, default=500.0)
    return parser


def _config_from_args(args: argparse.Namespace) -> ModelConfig:
    return {
        "num_params": float(args.num_params_b),
        "num_layers": int(args.num_layers),
        "hidden_size": int(args.hidden_size),
        "num_attention_heads": int(args.num_attention_heads),
        "num_key_value_heads": int(args.num_key_value_heads),
    }


def _print_result(quantization: str, result: CalculationResult) -> None:
    print(f"[{quantization}]")
    print(f"  max_context_length: {result['max_context_length']}")
    print(f"  model_weight_mb: {result['model_weight_mb']:.2f}")
    print(f"  kv_cache_mb: {result['kv_cache_mb']:.2f}")
    print(f"  total_memory_mb: {result['total_memory_mb']:.2f}")
    print(f"  fits_in_vram: {result['fits_in_vram']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = _config_from_args(args)
    profiler = ModelProfiler(config)
    kv_calc = KVCacheCalculator(config)
    calculator = VRAMCalculator(profiler, kv_calc, safety_margin_mb=args.safety_margin_mb)

    if args.quantization is None:
        results = calculator.suggest_optimal_config(args.free_vram_mb, args.backend)
        for quantization in ("FP16", "INT8", "INT4"):
            _print_result(quantization, results[quantization])
    else:
        quantization = cast(QuantizationType, args.quantization)
        max_context = calculator.find_max_context_length(
            args.free_vram_mb, quantization, args.backend
        )
        result: CalculationResult = {
            "max_context_length": max_context,
            "model_weight_mb": profiler.calculate_weight_size(quantization),
            "kv_cache_mb": kv_calc.calculate_kv_cache_size(max_context, quantization),
            "total_memory_mb": calculator.calculate_total_memory(
                max_context, quantization, args.backend
            ),
            "fits_in_vram": True,
        }
        _print_result(quantization, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

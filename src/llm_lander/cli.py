"""Minimal CLI for LLM-Lander calculations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import cast

from llm_lander.calculator import CalculationResult, VRAMCalculator
from llm_lander.gpu_scanner import GPUScanner
from llm_lander.hf_connector import HFConnector
from llm_lander.kv_cache import KVCacheCalculator
from llm_lander.model_profiler import ModelConfig, ModelProfiler, QuantizationType
from llm_lander.ollama_connector import OllamaConnector


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-Lander VRAM/Context calculator (MVP CLI)")
    parser.add_argument("--free-vram-mb", type=float, default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--model-source", choices=["manual", "hf", "ollama"], default="ollama")
    parser.add_argument("--model-name", type=str, default=None)
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

    metadata: dict[str, object] | None = None
    if args.free_vram_mb is None:
        scanner = GPUScanner()
        free_vram_mb = scanner.get_gpu_info(args.gpu_index)["free_vram_mb"]
    else:
        free_vram_mb = float(args.free_vram_mb)

    if args.model_source == "manual":
        config = _config_from_args(args)
    elif args.model_source == "hf":
        if not args.model_name:
            raise SystemExit("model-name is required for HF models")
        config = HFConnector().fetch_model_config(args.model_name)
    else:
        ollama = OllamaConnector()
        if not args.model_name:
            model_names = ollama.list_models()
            if not model_names:
                raise SystemExit("No Ollama models found")
            for name in model_names:
                config, metadata = ollama.fetch_model_profile(name)
                _run_single(config, free_vram_mb, args, name, metadata)
            return 0
        config, metadata = ollama.fetch_model_profile(args.model_name)
    _run_single(config, free_vram_mb, args, args.model_name, metadata)

    return 0


def _run_single(
    config: ModelConfig,
    free_vram_mb: float,
    args: argparse.Namespace,
    model_name: str | None,
    metadata: dict[str, object] | None,
) -> None:
    if model_name:
        quantization = _format_quantization(metadata)
        param_size = _format_param_size(metadata)
        print(f"\n=== {model_name}{quantization}{param_size} ===")
    print(f"free_vram_mb: {free_vram_mb:.2f}")
    profiler = ModelProfiler(config)
    kv_calc = KVCacheCalculator(config)
    calculator = VRAMCalculator(profiler, kv_calc, safety_margin_mb=args.safety_margin_mb)

    auto_quant = _resolve_quantization(metadata)
    if args.quantization is None and auto_quant is not None:
        _run_quantization(calculator, profiler, kv_calc, free_vram_mb, args, auto_quant)
        return

    if args.quantization is None:
        results = calculator.suggest_optimal_config(free_vram_mb, args.backend)
        for quantization in ("FP16", "INT8", "INT4"):
            _print_result(quantization, results[quantization])
    else:
        quantization = cast(QuantizationType, args.quantization)
        _run_quantization(calculator, profiler, kv_calc, free_vram_mb, args, quantization)


def _format_quantization(metadata: dict[str, object] | None) -> str:
    if not metadata:
        return ""
    details = metadata.get("details")
    if isinstance(details, dict):
        level = details.get("quantization_level")
        if isinstance(level, str) and level:
            return f" (quant={level})"
    return ""


def _format_param_size(metadata: dict[str, object] | None) -> str:
    if not metadata:
        return ""
    details = metadata.get("details")
    if isinstance(details, dict):
        size = details.get("parameter_size")
        if isinstance(size, str) and size:
            return f" (params={size})"
    return ""


def _resolve_quantization(metadata: dict[str, object] | None) -> QuantizationType | None:
    if not metadata:
        return None
    details = metadata.get("details")
    if not isinstance(details, dict):
        return None
    level = details.get("quantization_level")
    if not isinstance(level, str):
        return None
    text = level.upper()
    if "Q4" in text or "Q5" in text or "Q6" in text:
        return "INT4"
    if "Q8" in text:
        return "INT8"
    if "F16" in text or "FP16" in text:
        return "FP16"
    return None


def _run_quantization(
    calculator: VRAMCalculator,
    profiler: ModelProfiler,
    kv_calc: KVCacheCalculator,
    free_vram_mb: float,
    args: argparse.Namespace,
    quantization: QuantizationType,
) -> None:
    max_context = calculator.find_max_context_length(free_vram_mb, quantization, args.backend)
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


if __name__ == "__main__":
    raise SystemExit(main())

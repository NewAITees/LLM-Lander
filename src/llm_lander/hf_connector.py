"""Hugging Face Hub connector for model configs."""

from __future__ import annotations

import json
from typing import Any

from llm_lander.model_profiler import ModelConfig


def _load_hf_download() -> Any:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("huggingface_hub is not available") from exc
    return hf_hub_download


class HFConnector:
    """Fetch model configs from Hugging Face Hub."""

    def fetch_model_config(self, model_name: str) -> ModelConfig:
        if not model_name:
            raise ValueError("model_name is required")

        hf_hub_download = _load_hf_download()
        try:
            config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        except Exception as exc:
            raise ValueError(f"Failed to fetch config.json for {model_name}") from exc

        try:
            with open(config_path, encoding="utf-8") as handle:
                config_data = json.load(handle)
        except Exception as exc:
            raise ValueError("Failed to read config.json") from exc

        return {
            "num_params": self._estimate_params_from_config(config_data),
            "num_layers": int(config_data["num_hidden_layers"]),
            "hidden_size": int(config_data["hidden_size"]),
            "num_attention_heads": int(config_data["num_attention_heads"]),
            "num_key_value_heads": int(
                config_data.get("num_key_value_heads", config_data["num_attention_heads"])
            ),
        }

    def _estimate_params_from_config(self, config_data: dict[str, Any]) -> float:
        hidden_size = int(config_data["hidden_size"])
        num_layers = int(config_data["num_hidden_layers"])
        vocab_size = int(config_data.get("vocab_size", 32000))

        params = vocab_size * hidden_size
        params += num_layers * (4 * hidden_size * hidden_size + 8 * hidden_size * hidden_size)
        return params / 1e9

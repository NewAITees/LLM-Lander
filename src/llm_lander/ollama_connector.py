"""Ollama connector for resolving model configs."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any, TypedDict

from llm_lander.hf_connector import HFConnector
from llm_lander.model_profiler import ModelConfig


class OllamaModelInfo(TypedDict):
    """Parsed metadata from an Ollama Modelfile."""

    base_model: str
    parameters: dict[str, str]
    model_info: dict[str, Any]
    details: dict[str, Any]


class OllamaConnector:
    """Resolve Ollama models to base configs."""

    def __init__(self, hf_connector: HFConnector | None = None) -> None:
        self._hf = hf_connector or HFConnector()
        self._base_url = "http://localhost:11434"

    def fetch_model_config(
        self, model_name: str, info: OllamaModelInfo | None = None
    ) -> ModelConfig:
        if info is None:
            info = self._read_modelfile(model_name)
        base_model = info["base_model"]
        if base_model and self._looks_like_hf_repo(base_model):
            return self._hf.fetch_model_config(base_model)

        model_info = info["model_info"]
        config = self._config_from_model_info(model_info)
        if config is None:
            if not base_model:
                raise ValueError("Base model not found in Ollama Modelfile")
            raise ValueError(f"Unsupported base model reference: {base_model}")
        return config

    def fetch_model_profile(self, model_name: str) -> tuple[ModelConfig, dict[str, Any]]:
        info = self._read_modelfile(model_name)
        config = self.fetch_model_config(model_name, info)
        return config, {"details": info["details"], "model_info": info["model_info"]}

    def list_models(self) -> list[str]:
        data = self._curl_json(f"{self._base_url}/api/tags", None)
        models = data.get("models", [])
        if not isinstance(models, list):
            raise ValueError("Invalid tags response from Ollama")
        names: list[str] = []
        for entry in models:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                names.append(entry["name"])
        return names

    def _read_modelfile(self, model_name: str) -> OllamaModelInfo:
        if not model_name:
            raise ValueError("model_name is required")

        payload = {"model": model_name, "verbose": True}
        data = self._curl_json(f"{self._base_url}/api/show", payload)
        if "error" in data:
            raise ValueError(f"Ollama API error: {data['error']}")
        modelfile = data.get("modelfile", "")
        if not isinstance(modelfile, str):
            raise ValueError("Invalid modelfile response from Ollama")
        model_info = data.get("model_info", {})
        if not isinstance(model_info, dict):
            model_info = {}
        details = data.get("details", {})
        if not isinstance(details, dict):
            details = {}
        if "details.parameter_size" not in model_info and "parameter_size" in details:
            model_info["details.parameter_size"] = details["parameter_size"]

        base_model = ""
        parameters: dict[str, str] = {}
        for line in modelfile.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith("FROM "):
                base_model = stripped.split(" ", 1)[1].strip()
                continue
            match = re.match(r"PARAMETER\s+(\S+)\s+(.*)", stripped, re.IGNORECASE)
            if match:
                parameters[match.group(1)] = match.group(2).strip()

        return {
            "base_model": base_model,
            "parameters": parameters,
            "model_info": model_info,
            "details": details,
        }

    def _looks_like_hf_repo(self, name: str) -> bool:
        if "\\" in name or ":" in name:
            return False
        if "/" not in name:
            return False
        return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name))

    def _config_from_model_info(self, model_info: dict[str, Any]) -> ModelConfig | None:
        num_layers = self._get_int(model_info, ["llama.block_count", "block_count", "num_layers"])
        if num_layers is None:
            num_layers = self._get_int_by_suffix(model_info, ".block_count")

        hidden_size = self._get_int(
            model_info, ["llama.embedding_length", "embedding_length", "hidden_size"]
        )
        if hidden_size is None:
            hidden_size = self._get_int_by_suffix(model_info, ".embedding_length")

        num_heads = self._get_int(model_info, ["llama.head_count", "head_count", "num_heads"])
        if num_heads is None:
            num_heads = self._get_int_by_suffix(model_info, ".head_count")

        num_kv_heads = self._get_int(
            model_info, ["llama.head_count_kv", "head_count_kv", "num_key_value_heads"]
        )
        if num_kv_heads is None:
            num_kv_heads = self._get_int_by_suffix(model_info, ".head_count_kv")

        if num_layers is None or hidden_size is None or num_heads is None:
            return None
        if num_kv_heads is None:
            num_kv_heads = num_heads

        num_params_b = self._get_param_count_b(model_info, hidden_size, num_layers)
        return {
            "num_params": num_params_b,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
        }

    def _get_int(self, model_info: dict[str, Any], keys: list[str]) -> int | None:
        for key in keys:
            if key in model_info:
                value = model_info[key]
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _get_int_by_suffix(self, model_info: dict[str, Any], suffix: str) -> int | None:
        for key, value in model_info.items():
            if key.endswith(suffix):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _get_param_count_b(
        self, model_info: dict[str, Any], hidden_size: int, num_layers: int
    ) -> float:
        for key in (
            "general.parameter_count",
            "parameter_count",
            "llama.parameter_count",
            "details.parameter_size",
        ):
            if key in model_info:
                return self._parse_param_count(model_info[key])

        vocab_size = self._get_int(
            model_info, ["llama.vocab_size", "vocab_size", "general.vocab_size"]
        )
        if vocab_size is None:
            vocab_size = 32000
        params = vocab_size * hidden_size
        params += num_layers * (4 * hidden_size * hidden_size + 8 * hidden_size * hidden_size)
        return params / 1e9

    def _parse_param_count(self, value: Any) -> float:
        if isinstance(value, int | float):
            if value > 1e6:
                return float(value) / 1e9
            return float(value)
        if isinstance(value, str):
            text = value.strip().upper()
            try:
                if text.endswith("B"):
                    return float(text[:-1])
                if text.endswith("M"):
                    return float(text[:-1]) / 1000.0
                return float(text) / 1e9 if float(text) > 1e6 else float(text)
            except ValueError:
                return 0.0
        return 0.0

    def _curl_json(self, url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        cmd = ["curl", "-sS", url]
        if payload is not None:
            cmd.extend(
                [
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    json.dumps(payload),
                ]
            )
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("curl not found") from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError("Failed to query Ollama API") from exc

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON response from Ollama") from exc
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON response from Ollama")
        return data

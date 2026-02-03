from typing import Any

import pytest

from llm_lander.hf_connector import HFConnector
from llm_lander.model_profiler import ModelConfig
from llm_lander.ollama_connector import OllamaConnector


def test_fetch_model_config_via_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    modelfile = "FROM meta-llama/Llama-3-8B\nPARAMETER num_ctx 8192\n"

    def fake_curl_json(url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        assert url.endswith("/api/show")
        assert payload is not None
        assert payload["model"] == "llama3"
        assert payload["verbose"] is True
        return {
            "modelfile": modelfile,
            "model_info": {},
            "details": {"quantization_level": "Q4_K_M"},
        }

    def fake_fetch(self: Any, model_name: str) -> ModelConfig:  # noqa: ARG001
        return {
            "num_params": 7.0,
            "num_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }

    monkeypatch.setattr(OllamaConnector, "_curl_json", staticmethod(fake_curl_json))
    monkeypatch.setattr(HFConnector, "fetch_model_config", fake_fetch)

    connector = OllamaConnector()
    config = connector.fetch_model_config("llama3")
    assert config["num_layers"] == 32
    assert config["num_params"] == 7.0


def test_list_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_curl_json(url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        assert url.endswith("/api/tags")
        assert payload is None
        return {"models": [{"name": "llama3:latest"}, {"name": "mistral"}]}

    monkeypatch.setattr(OllamaConnector, "_curl_json", staticmethod(fake_curl_json))
    connector = OllamaConnector()
    assert connector.list_models() == ["llama3:latest", "mistral"]


def test_fetch_model_config_from_model_info(monkeypatch: pytest.MonkeyPatch) -> None:
    modelfile = "FROM C:\\\\Users\\\\perso\\\\.ollama\\\\models\\\\blobs\\\\sha256-abc\n"
    model_info = {
        "llama.block_count": 32,
        "llama.embedding_length": 4096,
        "llama.head_count": 32,
        "llama.head_count_kv": 8,
        "general.parameter_count": 7_000_000_000,
    }

    def fake_curl_json(_url: str, _payload: dict[str, Any] | None) -> dict[str, Any]:
        return {"modelfile": modelfile, "model_info": model_info}

    monkeypatch.setattr(OllamaConnector, "_curl_json", staticmethod(fake_curl_json))
    connector = OllamaConnector()
    config = connector.fetch_model_config("llama3")
    assert config["num_layers"] == 32
    assert config["num_key_value_heads"] == 8
    assert config["num_params"] == pytest.approx(7.0)

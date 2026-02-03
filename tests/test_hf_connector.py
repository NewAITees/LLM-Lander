import json
from pathlib import Path

import pytest

from llm_lander.hf_connector import HFConnector


def test_fetch_model_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fake_download(repo_id: str, filename: str) -> str:
        assert repo_id == "meta-llama/Llama-3-8B"
        assert filename == "config.json"
        return str(config_path)

    monkeypatch.setattr("llm_lander.hf_connector._load_hf_download", lambda: fake_download)

    connector = HFConnector()
    model_config = connector.fetch_model_config("meta-llama/Llama-3-8B")

    assert model_config["num_layers"] == 32
    assert model_config["hidden_size"] == 4096
    assert model_config["num_attention_heads"] == 32
    assert model_config["num_key_value_heads"] == 8
    assert model_config["num_params"] > 0

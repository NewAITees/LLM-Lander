import pytest

from llm_lander.cli import main


def test_cli_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--free-vram-mb",
            "12000",
            "--model-source",
            "manual",
            "--num-params-b",
            "7",
            "--num-layers",
            "32",
            "--hidden-size",
            "4096",
            "--num-attention-heads",
            "32",
            "--num-key-value-heads",
            "8",
            "--quantization",
            "FP16",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "max_context_length" in captured
    assert "total_memory_mb" in captured

# LLM-Lander

**LLM VRAM/Context Calculator** - Calculate maximum context length based on available VRAM for Large Language Models.

[日本語 READMEはこちら](README.md)

## Overview

LLM-Lander estimates the maximum context length for LLMs based on available VRAM. It accounts for model parameters, quantization, and KV cache usage to propose suitable configurations.

## Key Features

### Phase 1: Core Engine (Calculation Logic)
- **HW Direct Scan**: Real-time VRAM detection using `py3nvml`
- **Model Profiler**: Memory estimation based on params, layers, heads, GQA ratio
- **KV Cache Calculator**: Memory usage per context length (FP16/INT8/INT4)
- **Static Load Estimation**: Weights + activations + context total

### Phase 2: Intelligent Reverse Calculation
- **Reverse Search**: Fast estimation of max context length from free VRAM
- **Backend Profiles**: Consider overhead for llama.cpp, vLLM, AutoGPTQ, etc.

### Phase 3: Data & UI
- **Hugging Face Hub API**: Auto-load settings by model name
- **GUI / Web UI (planned)**: Streamlit interface
- **Visualization (planned)**: Plotly graphs for context vs VRAM

## Requirements

- Python 3.12+
- NVIDIA GPU (CUDA)
- Linux or Windows (Mac not supported due to lack of NVIDIA GPUs)

## Installation

```bash
git clone https://github.com/your-username/LLM-Lander.git
cd LLM-Lander
uv sync
```

## Usage

```bash
# Streamlit UI (planned)
uv run streamlit run src/llm_lander/app.py

# CLI output
python -m llm_lander
```

## Output Explanation (CLI)

```text
=== qwen3-coder:30b (quant=Q4_K_M) (params=30.5B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 267283
  model_weight_mb: 14558.85
  kv_cache_mb: 3132.22
  total_memory_mb: 19101.02
  fits_in_vram: True
```

- `=== ... ===`: Model name, quantization, parameter count
- `free_vram_mb`: Available VRAM at calculation time (MB)
- `[INT4]` / `[INT8]` / `[FP16]`: Results for each precision
- `max_context_length`: **Maximum token count that fits in available VRAM (theoretical upper bound)**
  - Not necessarily the model's built-in max context length
  - Real usable max can be lower due to model limits (RoPE/positional embedding) or backend constraints
- `model_weight_mb`: Estimated memory footprint of model weights (MB)
- `kv_cache_mb`: Estimated **KV cache** memory (Key/Value tensors stored for attention during inference) (MB)
- `total_memory_mb`: Total of weights + KV cache + overhead (MB)
- `fits_in_vram`: Whether the total fits within available VRAM

## Benchmark Results

- Japanese: `docs/benchmark_results.md`
- English: `docs/benchmark_results_en.md`

## Development

### Setup

```bash
uv sync
uv run pre-commit install
```

### Tests

```bash
uv run pytest
uv run pytest --cov
```

### Lint / Format

```bash
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .
uv run mypy src/llm_lander
```

## Tech Stack

| Role | Library |
|------|---------|
| GPU Info | py3nvml |
| Model Info | huggingface_hub |
| Numeric | numpy |
| UI (planned) | Streamlit |
| Visualization (planned) | Plotly |
| Lint/Format | Ruff |
| Type Check | Mypy |
| Test | Pytest |

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your proposal.

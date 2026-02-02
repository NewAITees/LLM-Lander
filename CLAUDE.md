# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-Lander is a VRAM/Context Calculator tool that calculates the maximum context length for Large Language Models based on available VRAM. It uses reverse calculation algorithms to determine optimal LLM configurations considering model parameters, quantization levels, and KV cache.

**Critical Requirements:**
- NVIDIA GPU required (CUDA-compatible)
- Python 3.12+
- This tool does NOT work on Mac (no NVIDIA GPU support)
- Linux/Windows only

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Install pre-commit hooks (REQUIRED before first commit)
uv run pre-commit install
```

### Running
```bash
# Launch Streamlit UI
uv run streamlit run src/llm_lander/app.py

# Test GPU detection
uv run python -c "import py3nvml.py3nvml as nvml; nvml.nvmlInit(); print('GPU OK')"
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/llm_lander --cov-report=html

# Run single test file
uv run pytest tests/test_gpu_scanner.py

# Run specific test
uv run pytest tests/test_model_profiler.py::test_calculate_weight_size_fp16
```

### Code Quality
```bash
# Lint check
uv run ruff check .

# Auto-fix
uv run ruff check --fix .

# Format
uv run ruff format .

# Type check
uv run mypy src/llm_lander

# Run all checks
uv run pre-commit run --all-files
```

## Architecture

LLM-Lander follows a three-layer architecture:

### 1. Presentation Layer (app.py)
Streamlit UI for user interaction and visualization with Plotly graphs.

### 2. Business Logic Layer
- **calculator.py**: Core reverse calculation algorithm using binary search (O(log n)) to find max context length from available VRAM
- **model_profiler.py**: Calculates model weight sizes based on parameter count and quantization type (FP32/FP16/INT8/INT4)
- **kv_cache.py**: Computes KV cache memory consumption per token, with GQA (Grouped Query Attention) support

### 3. Data Access Layer
- **gpu_scanner.py**: Real-time VRAM monitoring via py3nvml
- **hf_connector.py**: Fetches model config from Hugging Face Hub API

### Data Flow (Reverse Calculation)

```
User Input → GPU Scanner → HF Connector/Manual Input → Model Profiler
    → KV Cache Calculator → Binary Search Algorithm → Optimal Config Suggestion
```

**Key Algorithm**: Binary search over context length range [1, 1M] to find maximum fitting context:
```
total_memory = (model_weights + kv_cache + activation + safety_margin) * backend_overhead
```

### Critical Design Considerations

1. **GQA Support**: Modern models (Llama-3, etc.) use Grouped Query Attention where `num_key_value_heads < num_attention_heads`. Must use `num_key_value_heads` for KV cache calculation.

2. **Backend Overhead**: Different inference backends have different memory overhead:
   - llama.cpp: 1.05x
   - vLLM: 1.10x
   - AutoGPTQ: 1.08x
   - Transformers: 1.15x

3. **KV Cache Formula**:
   ```
   kv_cache_mb = 2 * num_layers * kv_heads * head_dim * context_length * bytes_per_param / (1024 * 1024)
   ```
   The factor `2` accounts for both Key and Value caches.

## Type System

All code uses strict type hints (mypy strict mode). Key type definitions:

```python
QuantizationType = Literal["FP32", "FP16", "INT8", "INT4"]
BackendType = Literal["llama.cpp", "vLLM", "AutoGPTQ", "Transformers"]

class ModelConfig(TypedDict):
    num_params: float  # in billions
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int  # For GQA

class GPUInfo(TypedDict):
    gpu_index: int
    gpu_name: str
    total_vram_mb: float
    used_vram_mb: float
    free_vram_mb: float
```

## Coding Standards

- **Type hints**: Mandatory on all functions/methods (strict mode)
- **Docstrings**: Google Style required
- **Line length**: 100 characters
- **Testing**: TDD approach, 80%+ coverage target
- **Pre-commit**: Blocks commits failing ruff/mypy checks

## Git Workflow

Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Test additions/modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements

Branch strategy: `main` (stable) ← `develop` (integration) ← `feature/*` or `fix/*`

## Common Pitfalls

1. **GPU Initialization**: Always call `nvml.nvmlInit()` before GPU operations and `nvml.nvmlShutdown()` in `__del__`
2. **GQA Models**: Use `num_key_value_heads` (not `num_attention_heads`) for KV cache calculations
3. **Memory Units**: Consistently use MB (not GB or bytes) for VRAM calculations
4. **Type Errors**: Run `uv run mypy src/llm_lander` before committing - pre-commit will enforce this

## Module Dependencies

```
app.py
├── gpu_scanner.py (py3nvml)
├── hf_connector.py (huggingface_hub)
├── model_profiler.py (numpy)
├── kv_cache.py → model_profiler.py
└── calculator.py → model_profiler.py + kv_cache.py
```

No circular dependencies. calculator.py is the highest-level business logic module.

## Testing Strategy

Mock external dependencies (py3nvml, huggingface_hub) in unit tests. For GPU tests without NVIDIA hardware, use:

```python
from unittest.mock import patch, MagicMock

@patch("py3nvml.py3nvml.nvmlInit")
@patch("py3nvml.py3nvml.nvmlDeviceGetCount")
def test_gpu_info_mock(mock_count, mock_init):
    mock_count.return_value = 1
    # ... test implementation
```

## Documentation

- `docs/requirements.md`: Detailed feature requirements and specifications
- `docs/architecture.md`: Full module designs with class definitions
- `docs/development.md`: Extended development guide with examples

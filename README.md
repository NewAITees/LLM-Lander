# LLM-Lander

**LLM VRAM/Context Calculator** - Calculate maximum context length based on available VRAM for Large Language Models.

[English README is here](README_EN.md)

## 概要

LLM-Landerは、利用可能なVRAMから大規模言語モデル（LLM）の最大コンテキスト長を逆算するツールです。モデルのパラメータ数、量子化ビット数、KVキャッシュなどを考慮して、最適な設定を提案します。

## 主な機能

### Phase 1: コア・エンジン（計算ロジック）
- **HWダイレクトスキャン**: `py3nvml` を使用したリアルタイムVRAM取得
- **モデル・プロファイラ**: パラメータ数、レイヤー数、Head数、GQA比率を考慮したメモリ計算
- **KVキャッシュ計算機**: コンテキスト長ごとの消費メモリ（FP16/INT8/INT4）を算出
- **静的ロード予測**: モデルの重み + 活性化 + コンテキストの合計を予測

### Phase 2: インテリジェント逆算機能（目玉機能）
- **逆算アルゴリズム**: 空きVRAMから最大コンテキスト長を高速算出
- **推論バックエンド・プロファイル**: llama.cpp, vLLM, AutoGPTQなどのオーバーヘッドを考慮

### Phase 3: データ連携・UI
- **Hugging Face Hub API連携**: モデル名から自動で設定を取得
- **GUI / Web UI（予定）**: Streamlitによる直感的なインターフェース
- **可視化グラフ（予定）**: Plotlyによるコンテキスト長とVRAM使用量の推移表示

## 必要要件

- Python 3.12以上
- NVIDIA GPU（CUDA対応）
- Linux または Windows（MacはNVIDIA GPU非搭載のため非対応）

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/your-username/LLM-Lander.git
cd LLM-Lander

# 依存関係のインストール（uvを使用）
uv sync
```

## 使い方

```bash
# Streamlit UIの起動（予定）
uv run streamlit run src/llm_lander/app.py
```

```bash
# CLI（計算結果の出力）
python -m llm_lander
```

## 出力の読み方（CLI）

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

- `=== ... ===`: モデル名、量子化形式、パラメータ数を表示
- `free_vram_mb`: 計算時点の空きVRAM（MB）
- `[INT4]` / `[INT8]` / `[FP16]`: 量子化精度ごとの結果
- `max_context_length`: **空きVRAMに収まる最大トークン数（理論上の上限）**
  - モデルが本来サポートする最大コンテキスト長と同義ではありません
  - 実際の上限はモデル仕様（RoPE/位置埋め込み）や推論バックエンドの制約で小さくなることがあります
- `model_weight_mb`: モデル重みの推定メモリ使用量（MB）
- `kv_cache_mb`: **KVキャッシュ**（注意機構のKey/Valueを保存する推論用メモリ）の推定使用量（MB）
- `total_memory_mb`: モデル重み + KVキャッシュ + オーバーヘッドの合計（MB）
- `fits_in_vram`: 合計が空きVRAMに収まるかどうか

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

## ベンチマーク結果

- 日本語: `docs/benchmark_results.md`
- English: `docs/benchmark_results_en.md`

## English (Quick Overview)

LLM-Lander is a VRAM/context calculator for LLMs. It estimates the maximum context length that fits in available VRAM, based on model parameters, quantization, and KV cache size.

### Requirements

- Python 3.12+
- NVIDIA GPU (CUDA)
- Linux or Windows (Mac is not supported due to lack of NVIDIA GPUs)

### Install

```bash
git clone https://github.com/your-username/LLM-Lander.git
cd LLM-Lander
uv sync
```

### Usage

```bash
# Streamlit UIの起動（予定）
uv run streamlit run src/llm_lander/app.py
python -m llm_lander
```

## 開発

### 開発環境のセットアップ

```bash
# 開発用依存関係を含めてインストール
uv sync

# pre-commitフックのインストール
uv run pre-commit install
```

### テスト

```bash
# テストの実行
uv run pytest

# カバレッジ付きテスト
uv run pytest --cov
```

### リント・フォーマット

```bash
# Ruffでリント
uv run ruff check .

# Ruffで自動修正
uv run ruff check --fix .

# Ruffでフォーマット
uv run ruff format .

# Mypyで型チェック
uv run mypy src/llm_lander
```

## 技術スタック

| 役割 | ライブラリ |
|------|-----------|
| GPU情報取得 | py3nvml |
| モデル情報取得 | huggingface_hub |
| 数値計算 | numpy |
| UI（予定） | Streamlit |
| 可視化（予定） | Plotly |
| リント/フォーマット | Ruff |
| 型チェック | Mypy |
| テスト | Pytest |

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

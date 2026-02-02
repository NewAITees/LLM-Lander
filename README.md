# LLM-Lander

**LLM VRAM/Context Calculator** - Calculate maximum context length based on available VRAM for Large Language Models.

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
- **GUI / Web UI**: Streamlitによる直感的なインターフェース
- **可視化グラフ**: Plotlyによるコンテキスト長とVRAM使用量の推移表示

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
# Streamlit UIの起動
uv run streamlit run src/llm_lander/app.py
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
| UI | Streamlit |
| 可視化 | Plotly |
| リント/フォーマット | Ruff |
| 型チェック | Mypy |
| テスト | Pytest |

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

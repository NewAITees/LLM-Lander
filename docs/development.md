# LLM-Lander 開発ガイド

## 1. 開発環境のセットアップ

### 1.1 前提条件

#### 必須要件
- Python 3.12以上
- NVIDIA GPU（CUDA対応）
- NVIDIA ドライバー
- Git
- uv（パッケージマネージャー）

#### 推奨環境
- Linux または Windows（MacはNVIDIA GPU非搭載のため非対応）
- 16GB以上のシステムメモリ
- VSCode または PyCharm

### 1.2 初期セットアップ

```bash
# 1. リポジトリのクローン
git clone https://github.com/your-username/LLM-Lander.git
cd LLM-Lander

# 2. Python 3.12のインストール（uvを使用）
uv python install 3.12

# 3. 依存関係のインストール
uv sync

# 4. pre-commitフックのインストール
uv run pre-commit install

# 5. GPU動作確認
nvidia-smi
uv run python -c "import py3nvml.py3nvml as nvml; nvml.nvmlInit(); print('GPU OK')"
```

### 1.3 エディタの設定

#### VSCode推奨拡張機能
- Python (Microsoft)
- Pylance
- Ruff
- Mypy Type Checker

#### VSCode設定（.vscode/settings.json）
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "strict",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

## 2. 開発ワークフロー

### 2.1 ブランチ戦略

```
main (安定版)
  └── develop (開発統合)
       ├── feature/gpu-scanner
       ├── feature/model-profiler
       ├── feature/kv-cache
       └── fix/bug-123
```

### 2.2 機能開発の流れ

#### ステップ1: ブランチ作成
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

#### ステップ2: TDD（テスト駆動開発）
```bash
# Red: 失敗するテストを書く
# tests/test_your_feature.py を作成

# テスト実行（失敗することを確認）
uv run pytest tests/test_your_feature.py

# Green: テストを通す最小限のコードを書く
# src/llm_lander/your_feature.py を実装

# テスト実行（成功することを確認）
uv run pytest tests/test_your_feature.py

# Refactor: コードを改善する
# リファクタリング...

# テスト実行（まだ成功することを確認）
uv run pytest tests/test_your_feature.py
```

#### ステップ3: コード品質チェック
```bash
# リント
uv run ruff check .

# 自動修正
uv run ruff check --fix .

# フォーマット
uv run ruff format .

# 型チェック
uv run mypy src/llm_lander

# すべてのチェックを一括実行
uv run pre-commit run --all-files
```

#### ステップ4: コミット
```bash
# 変更をステージング
git add src/llm_lander/your_feature.py tests/test_your_feature.py

# コミット（pre-commitが自動実行される）
git commit -m "feat: add your feature description"

# プッシュ
git push origin feature/your-feature-name
```

#### ステップ5: Pull Request
```bash
# GitHub CLIを使用（推奨）
gh pr create --title "feat: add your feature" --body "Description of your feature"

# または、GitHubのWebインターフェースから作成
```

### 2.3 コミットメッセージ規約

**フォーマット**: `<type>: <subject>`

**type一覧**:
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント
- `style`: コードスタイル
- `refactor`: リファクタリング
- `perf`: パフォーマンス改善
- `test`: テスト追加・修正
- `chore`: ビルド・ツール設定

**良い例**:
```
feat: add GPU VRAM scanner with py3nvml
fix: correct GQA head count calculation in KV cache
docs: update architecture documentation with diagrams
test: add unit tests for model profiler edge cases
refactor: extract common validation logic to utils
```

**悪い例**:
```
update code
fix bug
wip
...
```

## 3. テスト

### 3.1 テストの種類

#### ユニットテスト
個別のクラスや関数をテストする

```python
# tests/test_model_profiler.py
import pytest
from llm_lander.model_profiler import ModelProfiler, ModelConfig


def test_calculate_weight_size_fp16() -> None:
    """FP16でのモデル重みサイズ計算をテスト"""
    config: ModelConfig = {
        "num_params": 7.0,  # 7B
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
    }
    profiler = ModelProfiler(config)
    weight_mb = profiler.calculate_weight_size("FP16")

    # 7B * 2 bytes / (1024 * 1024) ≈ 13,351 MB
    assert abs(weight_mb - 13351.0) < 100.0


def test_calculate_weight_size_int4() -> None:
    """INT4でのモデル重みサイズ計算をテスト"""
    config: ModelConfig = {
        "num_params": 7.0,
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
    }
    profiler = ModelProfiler(config)
    weight_mb = profiler.calculate_weight_size("INT4")

    # 7B * 0.5 bytes / (1024 * 1024) ≈ 3,338 MB
    assert abs(weight_mb - 3338.0) < 50.0
```

#### 統合テスト
複数のモジュールの連携をテストする

```python
# tests/test_integration.py
def test_full_calculation_flow() -> None:
    """全体の計算フローをテスト"""
    # GPU情報取得
    scanner = GPUScanner()
    gpu_info = scanner.get_gpu_info(0)

    # モデル設定
    config: ModelConfig = {...}
    profiler = ModelProfiler(config)
    kv_calc = KVCacheCalculator(config)
    calculator = VRAMCalculator(profiler, kv_calc)

    # 最大コンテキスト長を計算
    max_context = calculator.find_max_context_length(
        gpu_info["free_vram_mb"], "FP16", "llama.cpp"
    )

    assert max_context > 0
```

#### モックテスト
外部依存をモックしてテストする

```python
# tests/test_gpu_scanner.py
from unittest.mock import patch, MagicMock
import pytest


@patch("py3nvml.py3nvml.nvmlInit")
@patch("py3nvml.py3nvml.nvmlDeviceGetCount")
@patch("py3nvml.py3nvml.nvmlDeviceGetHandleByIndex")
def test_get_gpu_info_mock(
    mock_handle: MagicMock,
    mock_count: MagicMock,
    mock_init: MagicMock,
) -> None:
    """モックを使用したGPU情報取得のテスト"""
    # モックの設定
    mock_count.return_value = 1
    # ... (詳細な設定)

    scanner = GPUScanner()
    gpu_info = scanner.get_gpu_info(0)

    assert gpu_info["gpu_index"] == 0
    assert gpu_info["total_vram_mb"] > 0
```

### 3.2 テストコマンド

```bash
# 全テスト実行
uv run pytest

# カバレッジ付きテスト
uv run pytest --cov=src/llm_lander --cov-report=html

# 特定のテストファイルのみ実行
uv run pytest tests/test_model_profiler.py

# 特定のテスト関数のみ実行
uv run pytest tests/test_model_profiler.py::test_calculate_weight_size_fp16

# 詳細出力
uv run pytest -v

# 失敗時に即座に停止
uv run pytest -x

# 並列実行（高速化）
uv run pytest -n auto
```

### 3.3 カバレッジ目標

- **ユニットテスト**: 80%以上
- **統合テスト**: 主要フロー100%
- **エッジケース**: 可能な限り網羅

## 4. コーディング規約

### 4.1 型ヒント

**必須**: すべての関数・メソッドに型ヒントを付ける

```python
# 良い例
def calculate_kv_cache_size(
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    context_length: int,
    bytes_per_param: float = 2.0,
) -> float:
    """KVキャッシュサイズを計算する"""
    return 2 * num_layers * kv_heads * head_dim * context_length * bytes_per_param


# 悪い例
def calculate_kv_cache_size(num_layers, kv_heads, head_dim, context_length, bytes_per_param=2.0):
    return 2 * num_layers * kv_heads * head_dim * context_length * bytes_per_param
```

### 4.2 docstring

**スタイル**: Google Style

```python
def find_max_context_length(
    free_vram_mb: float,
    quantization: QuantizationType = "FP16",
    backend: BackendType = "llama.cpp",
) -> int:
    """二分探索で最大コンテキスト長を求める

    Args:
        free_vram_mb: 空きVRAM（MB）
        quantization: 量子化タイプ（デフォルト: FP16）
        backend: 推論バックエンド（デフォルト: llama.cpp）

    Returns:
        最大コンテキスト長（トークン数）

    Raises:
        ValueError: 不正な入力値の場合

    Example:
        >>> calculator = VRAMCalculator(profiler, kv_calc)
        >>> max_context = calculator.find_max_context_length(20000, "FP16", "llama.cpp")
        >>> print(max_context)
        8192
    """
    # 実装...
```

### 4.3 命名規則

- **クラス**: PascalCase（例: `ModelProfiler`）
- **関数・変数**: snake_case（例: `calculate_weight_size`）
- **定数**: UPPER_SNAKE_CASE（例: `BYTES_PER_PARAM`）
- **プライベート**: 先頭にアンダースコア（例: `_internal_method`）

### 4.4 インポート順序

```python
# 1. 標準ライブラリ
import json
from typing import TypedDict, Literal

# 2. サードパーティライブラリ
import numpy as np
import streamlit as st

# 3. ローカルモジュール
from llm_lander.model_profiler import ModelProfiler
from llm_lander.kv_cache import KVCacheCalculator
```

### 4.5 エラーハンドリング

```python
# 良い例：具体的な例外を使用
def get_gpu_info(self, gpu_index: int = 0) -> GPUInfo:
    """GPU情報を取得する"""
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        # ...
    except nvml.NVMLError as e:
        raise RuntimeError(f"Failed to get GPU info: {e}") from e


# 悪い例：例外を握りつぶす
def get_gpu_info(self, gpu_index: int = 0) -> GPUInfo:
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        # ...
    except:
        pass  # 何もしない
```

## 5. パフォーマンス

### 5.1 最適化のポイント

#### 二分探索の使用
```python
# O(log n) の時間計算量
def find_max_context_length(self, free_vram_mb: float) -> int:
    left, right = 1, 1_000_000
    max_context = 0

    while left <= right:
        mid = (left + right) // 2
        # 計算...

    return max_context
```

#### キャッシュの活用
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_weight_size(self, quantization: QuantizationType) -> float:
    """重みサイズを計算（キャッシュ付き）"""
    # 計算...
```

### 5.2 パフォーマンス測定

```bash
# プロファイリング
uv run python -m cProfile -o profile.stats src/llm_lander/app.py

# プロファイル結果の表示
uv run python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

## 6. デバッグ

### 6.1 ログ出力

```python
import logging

logger = logging.getLogger(__name__)

def calculate_total_memory(self, context_length: int) -> float:
    """総メモリ使用量を計算する"""
    logger.debug(f"Calculating total memory for context_length={context_length}")
    # 計算...
    logger.info(f"Total memory: {total_mb:.2f} MB")
    return total_mb
```

### 6.2 デバッグコマンド

```bash
# デバッグモードで実行
uv run python -m pdb src/llm_lander/app.py

# ログレベルを変更
export LOG_LEVEL=DEBUG
uv run streamlit run src/llm_lander/app.py
```

## 7. CI/CD

### 7.1 GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v2
      - name: Set up Python
        run: uv python install 3.12
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest
      - name: Run linter
        run: uv run ruff check .
      - name: Run type checker
        run: uv run mypy src/llm_lander
```

## 8. トラブルシューティング

### 8.1 よくある問題

#### NVIDIA GPUが認識されない
```bash
# ドライバー確認
nvidia-smi

# NVMLテスト
uv run python -c "import py3nvml.py3nvml as nvml; nvml.nvmlInit(); print('OK')"
```

#### 依存関係の問題
```bash
# キャッシュクリア
uv clean

# 再インストール
uv sync --reinstall
```

#### pre-commitが失敗する
```bash
# 手動で修正
uv run ruff check --fix .
uv run ruff format .

# 再コミット
git add .
git commit -m "fix: resolve pre-commit issues"
```

## 9. リリース

### 9.1 バージョン管理

セマンティックバージョニング（SemVer）を使用:
- **MAJOR**: 破壊的変更
- **MINOR**: 後方互換性のある機能追加
- **PATCH**: 後方互換性のあるバグ修正

### 9.2 リリース手順

```bash
# 1. バージョンを更新
# src/llm_lander/__init__.py
__version__ = "0.2.0"

# pyproject.toml
version = "0.2.0"

# 2. CHANGELOG.mdを更新
# CHANGELOG.md に変更内容を記載

# 3. コミット
git add .
git commit -m "chore: bump version to 0.2.0"

# 4. タグ作成
git tag -a v0.2.0 -m "Release v0.2.0"

# 5. プッシュ
git push origin main --tags

# 6. GitHub Releaseを作成
gh release create v0.2.0 --title "v0.2.0" --notes "Release notes..."
```

## 10. 参考リソース

### 10.1 公式ドキュメント
- [Python公式](https://docs.python.org/3.12/)
- [Streamlit](https://docs.streamlit.io/)
- [Ruff](https://docs.astral.sh/ruff/)
- [Mypy](https://mypy.readthedocs.io/)
- [Pytest](https://docs.pytest.org/)

### 10.2 コーディング規約
- [PEP 8](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

### 10.3 テスト
- [Test-Driven Development by Example](https://www.amazon.com/Test-Driven-Development-Kent-Beck/dp/0321146530)

### 10.4 リファクタリング
- [Refactoring: Improving the Design of Existing Code](https://martinfowler.com/books/refactoring.html)

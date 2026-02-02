# LLM-Lander - AI Agents Development Guide

## プロジェクト構造

```
LLM-Lander/
├── src/
│   └── llm_lander/          # メインソースコード
│       ├── __init__.py      # パッケージ初期化
│       ├── app.py           # Streamlitアプリケーション（エントリーポイント）
│       ├── gpu_scanner.py   # GPU/VRAM情報取得（py3nvml使用）
│       ├── model_profiler.py # モデルプロファイリング
│       ├── kv_cache.py      # KVキャッシュ計算
│       ├── calculator.py    # 逆算アルゴリズム
│       └── hf_connector.py  # Hugging Face Hub連携
├── tests/                   # テストコード
│   ├── __init__.py
│   ├── test_gpu_scanner.py
│   ├── test_model_profiler.py
│   ├── test_kv_cache.py
│   └── test_calculator.py
├── docs/                    # ドキュメント
│   ├── requirements.md      # 詳細要件定義
│   ├── architecture.md      # アーキテクチャ設計
│   └── development.md       # 開発ガイド
├── pyproject.toml           # プロジェクト設定・依存関係管理
├── .pre-commit-config.yaml  # pre-commit設定
└── README.md                # プロジェクト概要
```

## 開発コマンド

### 基本操作
```bash
# 開発環境セットアップ
uv sync

# pre-commitフックのインストール
uv run pre-commit install

# アプリケーション起動
uv run streamlit run src/llm_lander/app.py
```

### テスト
```bash
# 全テスト実行
uv run pytest

# カバレッジ付きテスト
uv run pytest --cov=src/llm_lander --cov-report=html

# 特定のテストファイルのみ実行
uv run pytest tests/test_gpu_scanner.py

# 詳細出力
uv run pytest -v
```

### コード品質チェック
```bash
# リント（チェックのみ）
uv run ruff check .

# リント（自動修正）
uv run ruff check --fix .

# フォーマット
uv run ruff format .

# 型チェック
uv run mypy src/llm_lander

# すべてのチェックを一括実行
uv run pre-commit run --all-files
```

## コーディング規約

### Python
- **バージョン**: Python 3.12以上
- **スタイル**: Ruff（pycodestyle, pyflakes, isort, flake8-bugbear等）
- **型ヒント**: 必須（strict mode）
- **ドキュメント**: Google Style docstring

### 型ヒントの例
```python
def calculate_kv_cache_size(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    context_length: int,
    bytes_per_param: float = 2.0,
) -> float:
    """Calculate KV cache size in bytes.

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        context_length: Context length in tokens
        bytes_per_param: Bytes per parameter (2.0 for FP16, 1.0 for INT8)

    Returns:
        KV cache size in bytes
    """
    return 2 * num_layers * num_heads * head_dim * context_length * bytes_per_param
```

### docstringの例
```python
class ModelProfiler:
    """Profile LLM models for memory calculation.

    This class handles model profiling including parameter count,
    layer structure, and attention head configuration.

    Attributes:
        model_name: Name of the model
        num_params: Total number of parameters
        num_layers: Number of transformer layers
    """

    def __init__(self, model_name: str) -> None:
        """Initialize ModelProfiler.

        Args:
            model_name: Name or path of the model
        """
        self.model_name = model_name
```

## ブランチ戦略

- `main`: 安定版・デプロイ可能なコード
- `develop`: 開発統合ブランチ
- `feature/*`: 機能開発ブランチ
- `fix/*`: バグ修正ブランチ
- `hotfix/*`: 緊急修正ブランチ

### ワークフロー
```bash
# 新機能開発
git checkout -b feature/gpu-scanner
# 実装...
git add .
git commit -m "feat: add GPU VRAM scanner with py3nvml"
git push origin feature/gpu-scanner
# Pull Request作成
```

## コミットメッセージ規約

**フォーマット**: `<type>: <subject>`

**type一覧**:
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント
- `style`: コードスタイル（フォーマット等）
- `refactor`: リファクタリング
- `perf`: パフォーマンス改善
- `test`: テスト追加・修正
- `chore`: ビルド・ツール設定等

**例**:
```
feat: add KV cache calculator for FP16/INT8/INT4
fix: correct GQA head count calculation
docs: update architecture documentation
test: add unit tests for model profiler
```

## 依存関係

### メイン依存関係
- `py3nvml`: NVIDIA GPU情報取得
- `huggingface_hub`: モデル情報取得
- `numpy`: 数値計算
- `streamlit`: Web UI
- `plotly`: グラフ可視化

### 開発依存関係
- `ruff`: リント・フォーマット
- `mypy`: 型チェック
- `pytest`: テストフレームワーク
- `pytest-cov`: カバレッジ測定
- `pre-commit`: Git pre-commitフック

## 開発の進め方

### 1. 機能開発の流れ
1. issueを確認または作成
2. feature branchを作成
3. TDD（Test-Driven Development）で実装
4. pre-commitでコード品質を確保
5. Pull Requestを作成
6. レビュー後にmerge

### 2. TDDの実践
```bash
# Red: 失敗するテストを書く
uv run pytest tests/test_new_feature.py  # FAIL

# Green: テストを通す最小限のコードを書く
# 実装...
uv run pytest tests/test_new_feature.py  # PASS

# Refactor: コードを改善する
# リファクタリング...
uv run pytest tests/test_new_feature.py  # PASS
```

### 3. コミット前チェック
```bash
# 自動実行されるが、手動でも確認可能
uv run pre-commit run --all-files
```

## 注意事項

### NVIDIA GPUについて
- このアプリはNVIDIA GPU専用です
- MacではNVIDIA GPUがないため動作しません
- Linux/Windowsでの動作を想定しています

### pre-commitの重要性
- すべてのコミット前に自動でリント・フォーマット・型チェックが実行されます
- これにより、コード品質を一定に保ち、CI/CDでのエラーを防ぎます
- 必ず開発開始時に `uv run pre-commit install` を実行してください

### warningへの対応
- warning、info等のメッセージも積極的に対応してください
- deprecation warningは特に優先して対応
- 不要なログ出力は抑制設定を追加

## トラブルシューティング

### NVIDIA GPUが認識されない
```bash
# nvidia-smiで確認
nvidia-smi

# py3nvmlでテスト
uv run python -c "import py3nvml.py3nvml as nvml; nvml.nvmlInit(); print('OK')"
```

### 依存関係の問題
```bash
# キャッシュクリア
uv clean

# 再インストール
uv sync --reinstall
```

### pre-commitのスキップ（緊急時のみ）
```bash
# 推奨されませんが、緊急時はスキップ可能
git commit --no-verify
```

## 参考リソース

- [Python公式ドキュメント](https://docs.python.org/3.12/)
- [Streamlit公式ドキュメント](https://docs.streamlit.io/)
- [Ruff公式ドキュメント](https://docs.astral.sh/ruff/)
- [Mypy公式ドキュメント](https://mypy.readthedocs.io/)
- [NVIDIA NVML Documentation](https://developer.nvidia.com/nvidia-management-library-nvml)

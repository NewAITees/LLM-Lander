# LLM-Lander アーキテクチャ設計

## 1. システムアーキテクチャ

### 1.1 全体構成

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                   │
│                    （プレゼンテーション層）                    │
└────────────────┬───────────────────────────────┬─────────────┘
                 │                               │
                 ▼                               ▼
┌────────────────────────────┐  ┌─────────────────────────────┐
│   HF Connector Module      │  │   GPU Scanner Module        │
│   (hf_connector.py)        │  │   (gpu_scanner.py)          │
│                            │  │                             │
│ - Hugging Face Hub連携    │  │ - VRAM情報取得              │
│ - config.json取得          │  │ - GPU情報取得               │
└────────────────┬───────────┘  └──────────────┬──────────────┘
                 │                              │
                 └──────────┬───────────────────┘
                            ▼
          ┌─────────────────────────────────────┐
          │   Model Profiler Module             │
          │   (model_profiler.py)               │
          │                                     │
          │ - モデル重み計算                     │
          │ - 量子化考慮                        │
          │ - GQA対応                           │
          └──────────────┬──────────────────────┘
                         │
                         ▼
          ┌─────────────────────────────────────┐
          │   KV Cache Calculator Module        │
          │   (kv_cache.py)                     │
          │                                     │
          │ - KVキャッシュサイズ計算            │
          │ - 量子化KVキャッシュ対応            │
          └──────────────┬──────────────────────┘
                         │
                         ▼
          ┌─────────────────────────────────────┐
          │   Calculator Module                 │
          │   (calculator.py)                   │
          │                                     │
          │ - 逆算アルゴリズム                  │
          │ - 推論バックエンドプロファイル       │
          │ - 最適化提案                        │
          └─────────────────────────────────────┘
```

### 1.2 レイヤー構成

#### プレゼンテーション層
- **責務**: ユーザーインターフェース、入力バリデーション、結果表示
- **技術**: Streamlit, Plotly
- **ファイル**: `app.py`

#### ビジネスロジック層
- **責務**: コア計算ロジック、アルゴリズム実装
- **技術**: Python, NumPy
- **ファイル**: `calculator.py`, `model_profiler.py`, `kv_cache.py`

#### データアクセス層
- **責務**: 外部データ取得、ハードウェア情報取得
- **技術**: py3nvml, huggingface_hub
- **ファイル**: `gpu_scanner.py`, `hf_connector.py`

## 2. モジュール設計

### 2.1 gpu_scanner.py

**目的**: NVIDIA GPUの情報をリアルタイムで取得する

**クラス設計**:
```python
from typing import TypedDict
import py3nvml.py3nvml as nvml


class GPUInfo(TypedDict):
    """GPU情報の型定義"""
    gpu_index: int
    gpu_name: str
    total_vram_mb: float
    used_vram_mb: float
    free_vram_mb: float


class GPUScanner:
    """NVIDIA GPUの情報をスキャンするクラス"""

    def __init__(self) -> None:
        """NVMLを初期化する"""
        nvml.nvmlInit()

    def get_gpu_info(self, gpu_index: int = 0) -> GPUInfo:
        """指定されたGPUの情報を取得する

        Args:
            gpu_index: GPUのインデックス（デフォルト: 0）

        Returns:
            GPU情報の辞書

        Raises:
            RuntimeError: GPUが見つからない、またはドライバーエラー
        """
        ...

    def get_all_gpus(self) -> list[GPUInfo]:
        """すべてのGPUの情報を取得する

        Returns:
            GPU情報のリスト
        """
        ...

    def __del__(self) -> None:
        """NVMLをシャットダウンする"""
        nvml.nvmlShutdown()
```

**依存関係**:
- `py3nvml`: NVIDIA GPU情報取得

**エラーハンドリング**:
- NVIDIA GPUが見つからない: `RuntimeError`
- ドライバー未インストール: `RuntimeError`

### 2.2 model_profiler.py

**目的**: モデルのメモリフットプリントを計算する

**クラス設計**:
```python
from typing import Literal


QuantizationType = Literal["FP32", "FP16", "INT8", "INT4"]


class ModelConfig(TypedDict):
    """モデル設定の型定義"""
    num_params: float  # パラメータ数（単位: B）
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int  # GQA用


class ModelProfiler:
    """モデルのメモリプロファイリングを行うクラス"""

    BYTES_PER_PARAM: dict[QuantizationType, float] = {
        "FP32": 4.0,
        "FP16": 2.0,
        "INT8": 1.0,
        "INT4": 0.5,
    }

    def __init__(self, config: ModelConfig) -> None:
        """モデル設定で初期化する

        Args:
            config: モデル設定
        """
        self.config = config

    def calculate_weight_size(
        self, quantization: QuantizationType = "FP16"
    ) -> float:
        """モデルの重みサイズを計算する

        Args:
            quantization: 量子化タイプ（デフォルト: FP16）

        Returns:
            重みサイズ（MB）
        """
        num_params_total = self.config["num_params"] * 1e9
        bytes_per_param = self.BYTES_PER_PARAM[quantization]
        weight_size_mb = (num_params_total * bytes_per_param) / (1024 * 1024)
        return weight_size_mb

    def estimate_activation_memory(self, batch_size: int = 1) -> float:
        """活性化メモリを推定する

        Args:
            batch_size: バッチサイズ（デフォルト: 1）

        Returns:
            活性化メモリ（MB）
        """
        # 簡易的な推定式（実際はより複雑）
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]
        activation_mb = (batch_size * hidden_size * num_layers * 4) / (1024 * 1024)
        return activation_mb
```

**依存関係**:
- `numpy`: 数値計算

**計算式**:
```
weight_size_mb = (num_params * bytes_per_param) / (1024 * 1024)
```

### 2.3 kv_cache.py

**目的**: KVキャッシュのメモリサイズを計算する

**クラス設計**:
```python
class KVCacheCalculator:
    """KVキャッシュのメモリ計算を行うクラス"""

    def __init__(self, model_config: ModelConfig) -> None:
        """モデル設定で初期化する

        Args:
            model_config: モデル設定
        """
        self.config = model_config

    def calculate_kv_cache_size(
        self,
        context_length: int,
        quantization: QuantizationType = "FP16",
    ) -> float:
        """KVキャッシュサイズを計算する

        Args:
            context_length: コンテキスト長（トークン数）
            quantization: 量子化タイプ（デフォルト: FP16）

        Returns:
            KVキャッシュサイズ（MB）
        """
        num_layers = self.config["num_layers"]
        kv_heads = self.config["num_key_value_heads"]
        hidden_size = self.config["hidden_size"]
        num_attention_heads = self.config["num_attention_heads"]
        head_dim = hidden_size // num_attention_heads

        bytes_per_param = ModelProfiler.BYTES_PER_PARAM[quantization]

        # KVキャッシュ = 2 (Key + Value) * layers * kv_heads * head_dim * context_length * bytes
        kv_cache_bytes = (
            2 * num_layers * kv_heads * head_dim * context_length * bytes_per_param
        )
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)
        return kv_cache_mb

    def calculate_kv_cache_per_token(
        self, quantization: QuantizationType = "FP16"
    ) -> float:
        """1トークンあたりのKVキャッシュサイズを計算する

        Args:
            quantization: 量子化タイプ

        Returns:
            1トークンあたりのKVキャッシュサイズ（MB）
        """
        return self.calculate_kv_cache_size(context_length=1, quantization=quantization)
```

**依存関係**:
- `model_profiler.py`: `ModelConfig`, `QuantizationType`

**計算式**:
```
kv_cache_size = 2 * num_layers * kv_heads * head_dim * context_length * bytes_per_param
```

**注意点**:
- GQAの場合、`kv_heads < num_attention_heads`
- 最新モデル（Llama-3等）はGQAを採用

### 2.4 calculator.py

**目的**: 逆算アルゴリズムと最適化提案を実装する

**クラス設計**:
```python
from typing import Literal


BackendType = Literal["llama.cpp", "vLLM", "AutoGPTQ", "Transformers"]


class CalculationResult(TypedDict):
    """計算結果の型定義"""
    max_context_length: int
    model_weight_mb: float
    kv_cache_mb: float
    total_memory_mb: float
    fits_in_vram: bool


class VRAMCalculator:
    """VRAM計算と逆算を行うクラス"""

    BACKEND_OVERHEAD: dict[BackendType, float] = {
        "llama.cpp": 1.05,
        "vLLM": 1.10,
        "AutoGPTQ": 1.08,
        "Transformers": 1.15,
    }

    def __init__(
        self,
        model_profiler: ModelProfiler,
        kv_cache_calculator: KVCacheCalculator,
        safety_margin_mb: float = 500.0,
    ) -> None:
        """初期化

        Args:
            model_profiler: モデルプロファイラー
            kv_cache_calculator: KVキャッシュ計算機
            safety_margin_mb: 安全マージン（MB）
        """
        self.model_profiler = model_profiler
        self.kv_cache_calculator = kv_cache_calculator
        self.safety_margin_mb = safety_margin_mb

    def calculate_total_memory(
        self,
        context_length: int,
        quantization: QuantizationType = "FP16",
        backend: BackendType = "llama.cpp",
    ) -> float:
        """総メモリ使用量を計算する

        Args:
            context_length: コンテキスト長
            quantization: 量子化タイプ
            backend: 推論バックエンド

        Returns:
            総メモリ使用量（MB）
        """
        weight_mb = self.model_profiler.calculate_weight_size(quantization)
        kv_cache_mb = self.kv_cache_calculator.calculate_kv_cache_size(
            context_length, quantization
        )
        activation_mb = self.model_profiler.estimate_activation_memory()

        base_memory_mb = weight_mb + kv_cache_mb + activation_mb + self.safety_margin_mb
        total_memory_mb = base_memory_mb * self.BACKEND_OVERHEAD[backend]
        return total_memory_mb

    def find_max_context_length(
        self,
        free_vram_mb: float,
        quantization: QuantizationType = "FP16",
        backend: BackendType = "llama.cpp",
    ) -> int:
        """二分探索で最大コンテキスト長を求める

        Args:
            free_vram_mb: 空きVRAM（MB）
            quantization: 量子化タイプ
            backend: 推論バックエンド

        Returns:
            最大コンテキスト長（トークン数）
        """
        left, right = 1, 1_000_000
        max_context = 0

        while left <= right:
            mid = (left + right) // 2
            total_mb = self.calculate_total_memory(mid, quantization, backend)

            if total_mb <= free_vram_mb:
                max_context = mid
                left = mid + 1
            else:
                right = mid - 1

        return max_context

    def suggest_optimal_config(
        self, free_vram_mb: float, backend: BackendType = "llama.cpp"
    ) -> dict[str, CalculationResult]:
        """最適な設定を提案する

        Args:
            free_vram_mb: 空きVRAM（MB）
            backend: 推論バックエンド

        Returns:
            各量子化設定での計算結果
        """
        results = {}
        for quantization in ["FP16", "INT8", "INT4"]:
            max_context = self.find_max_context_length(
                free_vram_mb, quantization, backend  # type: ignore
            )
            weight_mb = self.model_profiler.calculate_weight_size(quantization)  # type: ignore
            kv_cache_mb = self.kv_cache_calculator.calculate_kv_cache_size(
                max_context, quantization  # type: ignore
            )
            total_mb = self.calculate_total_memory(max_context, quantization, backend)  # type: ignore

            results[quantization] = {
                "max_context_length": max_context,
                "model_weight_mb": weight_mb,
                "kv_cache_mb": kv_cache_mb,
                "total_memory_mb": total_mb,
                "fits_in_vram": total_mb <= free_vram_mb,
            }

        return results
```

**依存関係**:
- `model_profiler.py`: `ModelProfiler`
- `kv_cache.py`: `KVCacheCalculator`

**アルゴリズム**:
- 二分探索（Binary Search）で最大コンテキスト長を高速に算出
- 時間計算量: O(log n)

### 2.5 hf_connector.py

**目的**: Hugging Face Hubからモデル情報を取得する

**クラス設計**:
```python
from huggingface_hub import hf_hub_download
import json


class HFConnector:
    """Hugging Face Hub連携クラス"""

    def fetch_model_config(self, model_name: str) -> ModelConfig:
        """モデルのconfig.jsonを取得する

        Args:
            model_name: モデル名（例: "meta-llama/Llama-3-8B"）

        Returns:
            モデル設定

        Raises:
            ValueError: モデルが見つからない、またはconfig.jsonが取得できない
        """
        try:
            config_path = hf_hub_download(
                repo_id=model_name, filename="config.json"
            )
            with open(config_path, "r") as f:
                config_data = json.load(f)

            # ModelConfigに変換
            model_config: ModelConfig = {
                "num_params": self._estimate_params_from_config(config_data),
                "num_layers": config_data["num_hidden_layers"],
                "hidden_size": config_data["hidden_size"],
                "num_attention_heads": config_data["num_attention_heads"],
                "num_key_value_heads": config_data.get(
                    "num_key_value_heads", config_data["num_attention_heads"]
                ),
            }
            return model_config

        except Exception as e:
            raise ValueError(f"Failed to fetch model config: {e}")

    def _estimate_params_from_config(self, config_data: dict) -> float:
        """config.jsonからパラメータ数を推定する

        Args:
            config_data: config.jsonの内容

        Returns:
            パラメータ数（単位: B）
        """
        # 簡易的な推定（実際はより正確な計算が必要）
        hidden_size = config_data["hidden_size"]
        num_layers = config_data["num_hidden_layers"]
        vocab_size = config_data.get("vocab_size", 32000)

        # Transformerの基本的なパラメータ数の推定式
        # embedding + layers * (attention + ffn)
        params = vocab_size * hidden_size  # Embedding
        params += num_layers * (
            4 * hidden_size * hidden_size  # Attention (Q, K, V, O)
            + 8 * hidden_size * hidden_size  # FFN (up, down)
        )
        params_b = params / 1e9
        return params_b
```

**依存関係**:
- `huggingface_hub`: Hugging Face Hub API

**エラーハンドリング**:
- モデルが見つからない: `ValueError`
- config.jsonが取得できない: `ValueError`

### 2.6 app.py

**目的**: Streamlit UIを提供する

**構成**:
```python
import streamlit as st
import plotly.graph_objects as go


def main() -> None:
    """メイン関数"""
    st.title("LLM-Lander: VRAM/Context Calculator")

    # サイドバー: GPU情報
    with st.sidebar:
        st.header("GPU Information")
        # GPU情報の表示...

    # メインエリア: 入力フォーム
    st.header("Model Configuration")
    # モデル設定の入力...

    # 計算ボタン
    if st.button("Calculate"):
        # 計算実行...
        # 結果表示...
        # グラフ表示...


def display_gpu_info(gpu_info: GPUInfo) -> None:
    """GPU情報を表示する"""
    ...


def display_calculation_results(results: dict[str, CalculationResult]) -> None:
    """計算結果を表示する"""
    ...


def plot_vram_usage(
    context_lengths: list[int],
    vram_usage: dict[str, list[float]],
    free_vram_mb: float,
) -> go.Figure:
    """VRAM使用量のグラフを作成する"""
    ...


if __name__ == "__main__":
    main()
```

**依存関係**:
- すべてのモジュール
- `streamlit`: UI
- `plotly`: グラフ

## 3. データフロー

### 3.1 通常の計算フロー

```
1. ユーザー入力
   ↓
2. GPU情報取得 (gpu_scanner.py)
   ↓
3. モデル設定取得 (hf_connector.py または手動入力)
   ↓
4. モデルプロファイリング (model_profiler.py)
   ↓
5. KVキャッシュ計算 (kv_cache.py)
   ↓
6. 総メモリ計算 (calculator.py)
   ↓
7. 結果表示 (app.py)
```

### 3.2 逆算フロー

```
1. ユーザー入力
   ↓
2. GPU情報取得 (gpu_scanner.py)
   ↓
3. モデル設定取得 (hf_connector.py または手動入力)
   ↓
4. 逆算アルゴリズム実行 (calculator.py)
   ├─ 二分探索でmax_context_lengthを求める
   ├─ 各量子化設定で計算
   └─ 最適設定を提案
   ↓
5. 結果表示・グラフ表示 (app.py)
```

## 4. エラーハンドリング戦略

### 4.1 GPUエラー
- NVIDIA GPUが見つからない → エラーメッセージ + 終了
- ドライバー未インストール → 警告メッセージ + インストール方法の案内

### 4.2 Hugging Face Hubエラー
- モデルが見つからない → エラーメッセージ + 手動入力へ誘導
- ネットワークエラー → リトライまたは手動入力へ誘導

### 4.3 計算エラー
- 不正な入力値 → バリデーションエラー
- VRAMが不足 → 警告メッセージ + 推奨設定の提示

## 5. テスト戦略

### 5.1 ユニットテスト
- 各モジュールの個別テスト
- モックを使用した外部依存のテスト
- カバレッジ80%以上

### 5.2 統合テスト
- モジュール間の連携テスト
- エンドツーエンドのフローテスト

### 5.3 パフォーマンステスト
- 計算速度の測定
- メモリ使用量の測定

## 6. デプロイメント

### 6.1 ローカル実行
```bash
uv run streamlit run src/llm_lander/app.py
```

### 6.2 配布
- PyPI への公開（オプション）
- GitHub Releases での配布

## 7. 今後の拡張

### 7.1 マルチGPU対応
- Tensor Parallel の計算
- Pipeline Parallel の計算

### 7.2 高度な最適化
- Flash Attention の考慮
- LoRAアダプターのメモリ計算

### 7.3 データベース化
- モデルプロファイルのキャッシュ
- ベンチマーク結果の保存

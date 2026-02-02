# LLM-Lander 詳細要件定義

## 1. プロジェクト概要

### 1.1 目的
LLMを実行する際に、利用可能なVRAMから最大コンテキスト長を逆算し、最適な設定を提案するツールを開発する。

### 1.2 対象ユーザー
- LLMをローカルで実行するエンジニア・研究者
- NVIDIA GPUを搭載したマシンを使用するユーザー
- llama.cpp, vLLM, AutoGPTQなどの推論フレームワークを使用するユーザー

### 1.3 開発環境
- Python 3.12以上
- uv（パッケージマネージャー）
- NVIDIA GPU（CUDA対応）必須
- 対象OS: Linux, Windows（MacはNVIDIA GPU非搭載のため非対応）

## 2. 機能要件

### Phase 1: コア・エンジン（計算ロジック）

#### 2.1 HWダイレクトスキャン
**目的**: リアルタイムでVRAM情報を取得する

**要件**:
- py3nvmlを使用してNVIDIA GPUの情報を取得
- 取得する情報:
  - 総VRAM容量
  - 使用中VRAM
  - 空きVRAM
  - GPU名
  - GPUインデックス（複数GPU対応）
- エラーハンドリング:
  - NVIDIA GPUが見つからない場合のエラーメッセージ
  - ドライバーが未インストールの場合の警告

**入力**: なし（システムから自動取得）
**出力**: GPU情報の辞書（JSON形式）

```python
{
    "gpu_index": 0,
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "total_vram_mb": 24576,
    "used_vram_mb": 2048,
    "free_vram_mb": 22528
}
```

#### 2.2 モデル・プロファイラ
**目的**: モデルのメモリフットプリントを計算する

**要件**:
- モデルのパラメータ数から重みのサイズを計算
- 量子化ビット数の考慮:
  - FP16: 2 bytes/param
  - INT8: 1 byte/param
  - INT4: 0.5 bytes/param
  - FP32: 4 bytes/param
- 活性化メモリ（Activation）の推定
- GQA（Grouped Query Attention）の考慮

**入力**:
- モデルパラメータ数（例: 7B, 13B, 70B）
- 量子化ビット数（FP16, INT8, INT4）
- レイヤー数
- 隠れ層の次元数
- Attention Head数
- GQA比率

**出力**: モデルの重みサイズ（MB）

**計算式**:
```
weight_size_mb = (num_params * bits_per_param) / 8 / 1024 / 1024
```

#### 2.3 KVキャッシュ計算機
**目的**: コンテキスト長ごとのKVキャッシュメモリを計算する

**要件**:
- コンテキスト長に応じたKVキャッシュサイズの算出
- 量子化KVキャッシュへの対応（FP16, INT8, INT4）
- GQAの考慮（KV Headの数が異なる）

**入力**:
- レイヤー数（num_layers）
- Attention Head数（num_heads）
- Head次元数（head_dim）
- コンテキスト長（context_length）
- 量子化ビット数（bytes_per_param）

**出力**: KVキャッシュサイズ（MB）

**計算式**:
```
kv_cache_size = 2 * num_layers * kv_heads * head_dim * context_length * bytes_per_param
```

**注意**:
- GQAの場合、`kv_heads < num_heads`
- Llama-3等の最新モデルではGQAが採用されている

#### 2.4 静的ロード予測
**目的**: モデル全体のメモリ使用量を予測する

**要件**:
- モデルの重み + KVキャッシュ + 活性化メモリ の合計を計算
- 安全マージンの考慮（ユーザー設定可能、デフォルト500MB）
- 空きVRAMに収まるかの判定

**入力**:
- モデルの重みサイズ（MB）
- KVキャッシュサイズ（MB）
- 活性化メモリサイズ（MB）
- 安全マージン（MB）

**出力**: 総メモリ使用量（MB）、空きVRAMに収まるか（True/False）

### Phase 2: インテリジェント逆算機能（目玉機能）

#### 2.5 逆算アルゴリズム
**目的**: 空きVRAMから最大コンテキスト長を逆算する

**要件**:
- 二分探索による高速算出
- 複数の量子化設定での比較
- 最適解の提示

**入力**:
- 空きVRAM（MB）
- モデルの重みサイズ（MB）
- レイヤー数、Head数、Head次元数
- 量子化ビット数の候補リスト

**出力**:
- 各量子化設定での最大コンテキスト長
- 推奨設定（最もバランスが良い設定）

**アルゴリズム**:
```python
def binary_search_max_context(
    free_vram_mb: float,
    model_weight_mb: float,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    bytes_per_param: float,
    safety_margin_mb: float = 500.0,
) -> int:
    """二分探索で最大コンテキスト長を求める"""
    left, right = 1, 1_000_000  # 1 ~ 100万トークン
    max_context = 0

    while left <= right:
        mid = (left + right) // 2
        kv_cache_mb = calculate_kv_cache_size(
            num_layers, num_heads, head_dim, mid, bytes_per_param
        )
        total_mb = model_weight_mb + kv_cache_mb + safety_margin_mb

        if total_mb <= free_vram_mb:
            max_context = mid
            left = mid + 1
        else:
            right = mid - 1

    return max_context
```

#### 2.6 推論バックエンド・プロファイル
**目的**: 推論フレームワークごとのオーバーヘッドを考慮する

**要件**:
- 各バックエンドの管理領域を考慮した補正係数
- サポートバックエンド:
  - llama.cpp
  - vLLM
  - AutoGPTQ
  - Transformers

**入力**:
- バックエンド名
- 基本メモリ使用量（MB）

**出力**: 補正後のメモリ使用量（MB）

**補正係数（例）**:
```python
BACKEND_OVERHEAD = {
    "llama.cpp": 1.05,      # 5%のオーバーヘッド
    "vLLM": 1.10,           # 10%のオーバーヘッド
    "AutoGPTQ": 1.08,       # 8%のオーバーヘッド
    "Transformers": 1.15,   # 15%のオーバーヘッド
}
```

### Phase 3: データ連携・UI

#### 2.7 Hugging Face Hub API連携
**目的**: モデル名から自動で設定を取得する

**要件**:
- huggingface_hubライブラリを使用
- config.jsonから以下の情報を取得:
  - hidden_size（隠れ層の次元数）
  - num_hidden_layers（レイヤー数）
  - num_attention_heads（Attention Head数）
  - num_key_value_heads（KV Head数、GQA用）
- エラーハンドリング:
  - モデルが見つからない場合
  - config.jsonが取得できない場合

**入力**: モデル名（例: "meta-llama/Llama-3-8B"）
**出力**: モデル設定の辞書

#### 2.8 GUI / Web UI（Streamlit）
**目的**: 直感的なインターフェースを提供する

**要件**:
- **入力セクション**:
  - モデル選択（プリセット or 手動入力）
  - モデル名入力（Hugging Face Hub連携）
  - パラメータ数入力
  - 量子化ビット数選択
  - 安全マージン設定
  - 推論バックエンド選択
- **GPU情報セクション**:
  - GPU名、総VRAM、使用中VRAM、空きVRAMの表示
  - リフレッシュボタン
- **計算結果セクション**:
  - モデルの重みサイズ
  - 各コンテキスト長でのKVキャッシュサイズ
  - 最大コンテキスト長
  - 推奨設定
- **可視化セクション**:
  - コンテキスト長とVRAM使用量のグラフ

#### 2.9 可視化グラフ（Plotly）
**目的**: コンテキスト長とVRAM使用量の関係を視覚化する

**要件**:
- X軸: コンテキスト長（トークン数）
- Y軸: VRAM使用量（MB）
- 複数の量子化設定での比較（FP16, INT8, INT4）
- 空きVRAMのラインを表示
- インタラクティブな操作（ズーム、ホバー表示）

## 3. 非機能要件

### 3.1 パフォーマンス
- GPU情報取得: 100ms以内
- 逆算計算: 1秒以内
- Hugging Face Hub API: 3秒以内（ネットワーク依存）

### 3.2 ユーザビリティ
- 直感的なUI
- エラーメッセージの明確化
- デフォルト値の提供

### 3.3 保守性
- コードカバレッジ80%以上
- 型ヒント必須
- ドキュメント完備

### 3.4 セキュリティ
- ローカル実行のため、外部送信なし
- Hugging Face Hub APIのみ外部通信

## 4. 制約事項

### 4.1 ハードウェア制約
- NVIDIA GPU必須
- CUDA対応必須
- MacはNVIDIA GPU非搭載のため非対応

### 4.2 ソフトウェア制約
- Python 3.12以上
- NVIDIA ドライバーインストール必須

### 4.3 モデル制約
- Transformerベースのモデルのみ対応
- config.jsonが取得可能なモデルのみ自動設定可能

## 5. 今後の拡張

### 5.1 Phase 4: 高度な機能
- マルチGPU対応（Tensor Parallel）
- Flash Attention対応
- LoRAアダプターのメモリ計算
- 推論速度の予測

### 5.2 Phase 5: データベース化
- モデルプロファイルのキャッシュ
- ベンチマーク結果の保存
- ユーザー設定の保存

## 6. 用語集

| 用語 | 説明 |
|------|------|
| VRAM | Video RAM（GPU専用メモリ） |
| KVキャッシュ | Key-Valueキャッシュ（Attentionメカニズムで使用） |
| GQA | Grouped Query Attention（効率化されたAttention） |
| 量子化 | パラメータのビット数を削減する技術 |
| コンテキスト長 | モデルが一度に処理できるトークン数 |
| 活性化メモリ | 推論時の中間計算結果を保存するメモリ |

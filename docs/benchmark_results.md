# ベンチマーク結果（ローカル実行）

このドキュメントは、`python -m llm_lander` の実行結果を記録したものです。

- 計測日: 2026-02-03
- 実行環境: NVIDIA GPU（空きVRAM 19101.03 MB）
- 量子化: 表示されたモデルごとの設定に準拠

## 実行結果

```text
=== qwen3-coder:30b (quant=Q4_K_M) (params=30.5B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 267283
  model_weight_mb: 14558.85
  kv_cache_mb: 3132.22
  total_memory_mb: 19101.02
  fits_in_vram: True

=== glm-4.7-flash:latest (quant=Q4_K_M) (params=29.9B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 37325
  model_weight_mb: 14278.12
  kv_cache_mb: 3412.93
  total_memory_mb: 19101.00
  fits_in_vram: True

=== deepseek-coder-v2:16b (quant=Q4_0) (params=15.7B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 193456
  model_weight_mb: 7489.44
  kv_cache_mb: 10201.78
  total_memory_mb: 19101.00
  fits_in_vram: True

=== gemma3:12b (quant=Q4_K_M) (params=12.2B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 135162
  model_weight_mb: 5811.25
  kv_cache_mb: 11879.47
  total_memory_mb: 19101.00
  fits_in_vram: True

=== qwen3:8b (quant=Q4_K_M) (params=8.2B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 392113
  model_weight_mb: 3905.65
  kv_cache_mb: 13785.22
  total_memory_mb: 19101.00
  fits_in_vram: True

=== openbmb/minicpm-v4.5:latest (quant=Q4_0) (params=8.2B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 392134
  model_weight_mb: 3904.91
  kv_cache_mb: 13785.96
  total_memory_mb: 19101.01
  fits_in_vram: True

=== gpt-oss:20b (quant=MXFP4) (params=20.9B) ===
free_vram_mb: 19101.03
[FP16]
  max_context_length: 0
  model_weight_mb: 39891.73
  kv_cache_mb: 0.00
  total_memory_mb: 42411.60
  fits_in_vram: False
[INT8]
  max_context_length: 0
  model_weight_mb: 19945.87
  kv_cache_mb: 0.00
  total_memory_mb: 21468.44
  fits_in_vram: False
[INT4]
  max_context_length: 936711
  model_weight_mb: 9972.93
  kv_cache_mb: 7718.26
  total_memory_mb: 19101.03
  fits_in_vram: True

=== llava:7b (quant=Q4_0) (params=7B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 455610
  model_weight_mb: 3453.13
  kv_cache_mb: 14237.81
  total_memory_mb: 19101.01
  fits_in_vram: True

=== deepseek-r1:8b (quant=Q4_K_M) (params=8.2B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 392113
  model_weight_mb: 3905.65
  kv_cache_mb: 13785.22
  total_memory_mb: 19101.00
  fits_in_vram: True

=== qwen3:14b (quant=Q4_K_M) (params=14.8B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 272604
  model_weight_mb: 7042.08
  kv_cache_mb: 10648.59
  total_memory_mb: 19101.03
  fits_in_vram: True

=== llama3.1:8b (quant=Q4_K_M) (params=8.0B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 443578
  model_weight_mb: 3829.13
  kv_cache_mb: 13861.81
  total_memory_mb: 19101.01
  fits_in_vram: True

=== qwen3:14b-16k (quant=Q4_K_M) (params=14.8B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 272604
  model_weight_mb: 7042.08
  kv_cache_mb: 10648.59
  total_memory_mb: 19101.03
  fits_in_vram: True

=== qwen3:8b-16k (quant=Q4_K_M) (params=8.2B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 392113
  model_weight_mb: 3905.65
  kv_cache_mb: 13785.22
  total_memory_mb: 19101.00
  fits_in_vram: True

=== deepseek-r1:14b (quant=Q4_K_M) (params=14.8B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 227149
  model_weight_mb: 7042.90
  kv_cache_mb: 10647.61
  total_memory_mb: 19101.02
  fits_in_vram: True

=== gemma3:27b (quant=Q4_K_M) (params=27.4B) ===
free_vram_mb: 19101.03
[INT4]
  max_context_length: 29002
  model_weight_mb: 13080.63
  kv_cache_mb: 4609.45
  total_memory_mb: 19100.91
  fits_in_vram: True
```

## 補足: max_context_length の意味

- max_context_length は、**この環境の空きVRAMに収まる最大のトークン数**を意味します。
- これは「モデルが本来サポートする最大コンテキスト長（学習時の上限）」と同義ではありません。
- 実際の上限は、モデルの仕様（RoPE/位置埋め込みの上限）や推論バックエンドの制約により小さくなる場合があります。

つまり、ここでの max_context_length は **VRAM制約ベースの理論上の上限**です。

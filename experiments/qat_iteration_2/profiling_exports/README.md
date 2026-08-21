# QAT / PTQ trtexec profiling exports

> **Note:** the three per-kernel profile JSONs (`qat_batch32_`, `ptq_int8_`, `fp16_per_kernel_profile.json`) now live in `../Per_kernel_json_file/`. The `*_layer_info.json` / `*_layers.json` and analysis files remain here.


Raw profiling JSONs for the production TensorRT engines, for inspection/download.
Generated with **trtexec 10.16.1.11** (`--exportProfile` + `--exportLayerInfo`,
`--profilingVerbosity=detailed --separateProfileRun --noDataTransfers`), on an
exclusive idle A100-SXM4-80GB (Geneva). Median GPU compute latencies this session:
**QAT batch32 = 1.200 ms**, **PTQ INT8 = 1.072 ms** (batch=1, no CUDA graph).

## QAT batch32 engine (the primary files — mAP50-95 0.7797)

Engine: `experiments/qat_iteration_2/rebuild_fp16_opt5/V2_batch_32.engine`
(QAT-trained, batch=32 sweep winner, rebuilt with FP16 + opt-level-5).

| file | what it is |
|---|---|
| `qat_batch32_per_kernel_profile.json` | **trtexec `--exportProfile`** — per-kernel timing (245 kernels): `name`, `timeMs` (total), `averageMs` (per-inference), `medianMs`, `percentage`. First element is `{"count": N}`. |
| `qat_batch32_layer_info.json` | **trtexec `--exportLayerInfo`** — per-layer structure: `Name`, `LayerType`, `Inputs`/`Outputs` with `Format/Datatype` (this is where precision — INT8/FP16/FP32 — is read from). |
| `qat_batch32_per_layer_ranked.json` | our parse: every kernel ranked by time, with region (backbone/neck/attention/head/NMS) + precision + ms + %. |
| `qat_batch32_region_map_trtexec.json` | region-level latency aggregation (from the trtexec profile). |
| `qat_batch32_region_map_pyprofiler.json` | region-level aggregation from the TensorRT Python `IProfiler` (cross-check of the above). |

## PTQ INT8 engine (for comparison — mAP50-95 0.7564)

Engine: `experiments/qat_iteration_2/ptq_baseline/best_int8_fp16.engine`
(post-training quantization, INT8 + FP16 fallback).

| file | what it is |
|---|---|
| `ptq_int8_per_kernel_profile.json` | trtexec `--exportProfile` — per-kernel timing (189 kernels). |
| `ptq_int8_layer_info.json` | trtexec `--exportLayerInfo` — per-layer structure/precision. |

## Comparison

| file | what it is |
|---|---|
| `qat_vs_ptq_deep_compare.json` | machine-readable QAT-vs-PTQ summary (per-region + per-precision time/job splits). Headline: QAT's +0.127 ms vs PTQ is the SiLU activations split out of conv fusion (86 standalone SiLU kernels vs PTQ's 21). |

## Notes for readers

- **`averageMs` is per-inference; `timeMs` is the total over all profiled iterations.** To reproduce a region % , sum `averageMs` over the kernels in that region / sum over all kernels.
- The per-kernel `averageMs` are **instrumented** (profiling adds ~1.58× overhead): they sum to ~1.91 ms for QAT, which scales to the real 1.200 ms median. Shares are unaffected by the scaling.
- Precision is **not** a field on a layer — it is read from each layer's output `Format/Datatype` in the `*_layer_info.json`.
- `name` fields show TensorRT's fusions explicitly, e.g. `.../conv/Conv + PWN(Sigmoid, Mul)` = a fused conv+SiLU kernel.

## FP16 engine (same-architecture, NO Q/DQ baseline — mAP50-95 0.7748)

Engine: `models/yolo26n_sanoscience_full_left/2_fp16/best_fp16.engine`
Median GPU compute this session: **1.127 ms** (batch=1, no CUDA graph). Same trtexec
flags as the QAT/PTQ profiles.

| file | what it is |
|---|---|
| `fp16_per_kernel_profile.json` | trtexec `--exportProfile` — per-kernel timing (231 kernels). Same schema as QAT/PTQ. |
| `fp16_layers.json` | trtexec `--exportLayerInfo` — per-layer structure/precision (FP16). |

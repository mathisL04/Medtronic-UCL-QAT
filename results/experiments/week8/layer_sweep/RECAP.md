# Week 8 — per-layer QAT sweep: final recap

YOLO26n surgical-tool detector, INT8 QAT one layer at a time, exported to ONNX and
built as explicit-Q/DQ TensorRT engines.

Run finished **2026-08-21**. Train pass ended 02:57:55; deploy pass ran 02:59:55 → 05:01:39
on an exclusive GPU, exit 0 (`SWEEP COMPLETE`).

## 1. Where the models stand

All four rows below share **one eval harness** — full val (6,449 images), `conf=0.001`,
`max_dets=300`, imgsz 640, `pycocotools COCOeval bbox` — so they are directly comparable.

| model | precision | mAP50 | mAP50-95 |
|---|---|---|---|
| Original full-model baseline | FP32 engine | 0.9325 | **0.7747** |
| Frozen-head retrain | FP16 engine | 0.7419 | 0.5313 |
| V1 0-iteration QAT (calibrate only, no training) | INT8 | 0.6906 | **0.4892** |
| Best single-layer QAT (`model.8`) | INT8 | 0.8591 | **0.6675** |

Calibration-only INT8 costs **−0.0421** mAP50-95 against the frozen-head FP16 engine
(0.5313 → 0.4892). Six epochs of QAT on the single best layer more than repays that:
`model.8` lands **+0.1362 above** the FP16 frozen baseline and recovers **62.4%** of the
V1 → FP32 gap. Single-layer QAT does not reach the FP32 baseline, and was never expected to.

## 2. Per-layer results — 18 layers, ranked

`%gap` = fraction of the V1(0.4892) → FP32(0.7747) gap recovered.
`vs FP16` = delta against the frozen-head FP16 engine (0.5313).

| rank | layer | mAP50 | mAP50-95 | Δ vs V1 | %gap | vs FP16 | graph ms | epochs |
|---|---|---|---|---|---|---|---|---|
| 1 | model.8 | 0.8591 | 0.6675 | +0.1785 | 62.4% | +0.1362 | 0.7656 | 6 |
| 2 | model.6 | 0.8511 | 0.6525 | +0.1635 | 57.2% | +0.1212 | 0.7592 | 6 |
| 3 | model.13 | 0.8387 | 0.6518 | +0.1628 | 57.0% | +0.1206 | 0.7445 | 6 |
| 4 | model.10 | 0.8241 | 0.6301 | +0.1411 | 49.4% | +0.0989 | 0.7521 | 6 |
| 5 | model.4 | 0.7947 | 0.6205 | +0.1315 | 46.0% | +0.0893 | 0.7615 | 6 |
| 6 | model.7 | 0.8097 | 0.6168 | +0.1278 | 44.7% | +0.0855 | 0.7553 | 5 |
| 7 | model.9 | 0.8103 | 0.6065 | +0.1175 | 41.1% | +0.0752 | 0.7533 | 6 |
| 8 | model.5 | 0.7823 | 0.5994 | +0.1104 | 38.6% | +0.0681 | 0.7443 | 5 |
| 9 | model.16 | 0.7758 | 0.5846 | +0.0956 | 33.4% | +0.0534 | 0.7396 | 6 |
| 10 | model.3 | 0.7657 | 0.5769 | +0.0879 | 30.7% | +0.0456 | 0.7568 | 6 |
| 11 | model.19 | 0.7694 | 0.5685 | +0.0795 | 27.8% | +0.0373 | 0.7544 | 6 |
| 12 | model.2 | 0.7516 | 0.5568 | +0.0678 | 23.7% | +0.0255 | 0.7580 | 4 |
| 13 | model.17 | 0.7197 | 0.5390 | +0.0500 | 17.4% | +0.0077 | 0.7427 | 6 |
| 14 | model.1 | 0.7341 | 0.5343 | +0.0453 | 15.8% | +0.0030 | 0.7582 | 6 |
| 15 | model.23 | 0.7172 | 0.5025 | +0.0135 | 4.6% | −0.0288 | 0.7585 | 5 |
| 16 | model.0 | 0.7038 | 0.4982 | +0.0092 | 3.1% | −0.0331 | 0.7437 | 5 |
| 17 | model.20 | 0.6965 | 0.4949 | +0.0059 | 2.0% | −0.0364 | 0.7413 | 6 |
| 18 | model.22 | 0.6949 | 0.4874 | −0.0016 | −0.7% | −0.0439 | 0.7566 | 6 |

Skipped, `no_trainable_params`: `model.11`, `model.12`, `model.14`, `model.15`,
`model.18`, `model.21` — the Concat/Upsample nodes. Expected, not a failure.

**The trend the sweep existed to produce:** mid/late backbone (4–10, 13) carries essentially
all the recoverable accuracy. The stem (`model.0`) and the head (20, 22, 23) buy nothing —
`model.22` is fractionally negative. Layers 15–18 in the ranking sit *below* the FP16
frozen baseline, i.e. quantizing them costs more than training them back recovers.

## 3. Latency — no signal, do not rank on it

| metric | min | max | spread |
|---|---|---|---|
| CUDA-graph (ms) | 0.7396 | 0.7656 | 3.5% |
| no-graph (ms) | 0.9532 | 1.0026 | 5.2% |
| FPS (from graph time) | 1306 | 1352 | — |
| kernel count | 235 | 256 | — |
| engine size (MB) | 4.50 | 4.62 | — |

The spread is uncorrelated with which layer was quantized. At this precision the latency
axis does not discriminate between layers. All 18 were measured in one exclusive-GPU window,
so the readings are internally comparable — there is simply nothing to see.

**Do not compare these to the Section 1 rows.** The sweep was measured GPU-only on an H100;
the frozen/V1 engines were measured as full-pipeline totals on an A100 (median total 2.858 ms
/ 3.030 ms, ~350 / ~330 FPS). Different box and different harness — the accuracy axis is
comparable across all runs, the latency axis is not.

## 4. Toolchain

| stage | detail |
|---|---|
| QAT | ModelOpt 0.33.1, torch 2.7.0+cu128, `INT8_DEFAULT_CFG`, 608 quantizers inserted / 253 with scales |
| calibration | 128 frames, seed 42, `max` method |
| export | ONNX opset 17, batch 1, 640x640, 207 QuantizeLinear / 207 DequantizeLinear |
| build | TensorRT 10.16.1.11, explicit Q/DQ (no calibrator), `OPT_LEVEL=3`, 8 GB workspace, ~268 s/layer |
| training budget | 6 epochs, patience 3, uniform across layers |

## 5. Caveats

1. **Budget was not perfectly uniform.** Early stopping cut some layers to 4–5 epochs
   (`model.2` to 4, `model.5`/`7`/`23`/`0` to 5). `best_epoch` is 0-indexed, so `model.2`'s
   best was its *first* epoch and it degraded after — a short early-peaking run, not a
   starved one. The effect is small but it does slightly confound layer-vs-layer comparison.
2. **Single-layer QAT is a ranking experiment, not a deployment candidate.** The best layer
   still sits 0.107 mAP50-95 below the FP32 baseline.
3. **Engines have been cleared** to reclaim disk. Every engine's sha256 and full build
   configuration is preserved in the committed `*.engine.provenance.json` sidecars, and each
   is rebuildable from the retained `best_qat.onnx`.

## 6. Provenance

Committed alongside this file: `results_master_malmo_h100.csv` (source of every number
above), per-layer `metrics.json`, `train/results.csv`, `train/args.yaml`,
`best_qat.onnx.provenance.json`, `engine_int8_h100.engine.provenance.json`,
`*.map_full.json`, and the per-layer train/deploy logs.

### Exact code that produced these numbers

`main-consolidate` was later merged with a documentation refactor of the script tree, so
the scripts as they stand today are *formatted* differently from the ones that ran. The
behaviour is unchanged, and every env knob the sweep relies on was carried across, but for
byte-exact provenance use the pre-merge commit:

| what | where |
|---|---|
| tree as it ran the sweep | `bb32730` |
| `train_qat.py` as it ran | `git show bb32730:train/train_qat.py` (it lived at `train/` then; it is at `scripts/train/` now) |
| `build_tensorrt_int8_qdq.py` as it ran | `git show bb32730:scripts/tensorrt/build_tensorrt_int8_qdq.py` |
| sweep runners | `run_layer_malmo.sh`, `run_sweep_malmo.sh`, `queue_deploy_malmo.sh`, `make_metrics_malmo.py` |

The ONNX and engine sha256 in each `*.provenance.json` are unaffected by the refactor —
they identify the exact artifacts that were measured.

`results_master.csv` and `results_master_a100_geneva.csv` are **not** results — they are the
abandoned partial geneva A100 run, mostly empty rows. They are deliberately untracked.

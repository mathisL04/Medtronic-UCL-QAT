# Documentation

Stage-by-stage record of the project, in the order the work happened. Each document
covers one stage: what was run, on which machine, what came out, and what the result
does and does not license.

**New here?** Read `00`, then the root [`README.md`](../README.md) for the results
summary, then whichever stage you need.

| | Document | Stage | Covers |
|---|---|---|---|
| 00 | [Environment and access](00_environment_and_access.md) | — | The three machines, ssh, the four venvs, the idle-GPU rule, the devcontainer caveat |
| 01 | [Dataset, auto-labelling, training](01_dataset_labelling_training.md) | 1 | Open-H Sanoscience videos → segmentation-derived YOLO labels → YOLO26n baseline |
| 02 | [ONNX export](02_onnx_export.md) | 2 | Export, PyTorch/ONNX parity, the NMS-free output |
| 03 | [TensorRT FP32](03_tensorrt_fp32.md) | 3 | V2 baseline engine, the TF32 decision |
| 04 | [TensorRT FP16](04_tensorrt_fp16.md) | 4 | V3, what actually reduces to FP16, why parity "fails" correctly |
| 05 | [TensorRT INT8 PTQ](05_tensorrt_int8_ptq.md) | 5 | V4, entropy calibration, implicit vs explicit quantisation |
| 06 | [Quantisation-aware training](06_qat.md) | 6–7 | V5/V6, iteration-2 OFAT sweep, why QAT's engine was slower and how the build recovered it |
| 07 | [PyTorch-side latency](07_pytorch_latency.md) | — | Raw eager-mode cost, why `.half()` buys nothing, the FP32 baseline |
| 08 | [Week 8: layer sensitivity](08_week8_layer_sensitivity.md) | 8 | Frozen baseline, per-layer INT8 QAT sweep, what moves accuracy vs latency |

## The pipeline

```text
Open-H Sanoscience video ──▶ segmentation-derived YOLO labels ──▶ YOLO26n training
                                                                        │  docs/01
                                                                        ▼
                                                     ONNX export ──────────  docs/02
                                                                        │
                              ┌─────────────┬─────────────┬─────────────┤
                              ▼             ▼             ▼             ▼
                            FP32          FP16       INT8 PTQ      INT8 QAT
                          docs/03       docs/04       docs/05       docs/06
                                                                        │
                                                        iteration 2 ────┤  docs/06
                                                                        │
                                        week 8: which layers cost what ─┘  docs/08
```

## Where the numbers live

| | |
|---|---|
| `models/` | artifacts, in ladder order: `0_baseline_pytorch` → `1_fp32` → `2_fp16` → `3_int8_ptq` → `4_qat_iteration1` → `5_qat_iteration2` |
| `reports/` | everything measured, in the same order |
| `experiments/` | the runs themselves: `qat_iteration_2/`, `week8/`, `fusion_demo/` |

Engines, ONNX and checkpoints are gitignored; their `*.provenance.json` sidecars are
committed and carry the sha256 plus the full build configuration, so a cleared
artifact stays identifiable and rebuildable.

**Provenance files and result JSONs record paths as they were when the measurement
ran.** They are deliberately not rewritten when the repository is reorganised: they
are historical records, and the sha256 in each is the join key.

## Conventions that apply throughout

- **mAP is measured at `conf=0.001`, never at a deployment threshold.** `conf=0.25`
  truncates the precision-recall curve and under-reports mAP50 by ~5 points. See
  docs/01, "Final Training Results".
- **Latency is measured on a verified-idle GPU**, batch=1, median, with the device
  pinned and recorded. Any figure that was not says so. See docs/00.
- **Latency is not comparable across machines.** Cork V100, Geneva A100 and malmo
  H100 all appear in this repository; each figure names its GPU.

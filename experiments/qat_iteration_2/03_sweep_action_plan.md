# QAT Hyperparameter Sweep — Action Plan (design — for sign-off)

Scientific design for the QAT sweep phase. **One-factor-at-a-time (OFAT)**: fix everything at a
center config, vary one axis, reset, next. Each run is fully monitored (see `02_monitoring_scheme.md`)
and compared against the locked PTQ baseline. **Run nothing yet** — this is for review.

---

## 1. Design principles
- **OFAT**, not grid — isolates each hyperparameter's effect so any Δaccuracy is attributable to one
  variable (a grid would confound and cost 5–10× the runs).
- **Fixed center = the V6 config** (the QAT run we already validated). Every sweep varies ONE axis
  from this center; the center run is shared across all sweeps (run once).
- **Controls held constant** every run: `CALIB_SEED=42`, torch seed, train/val split, warm-start
  source, `WORKERS=0`, `amp=False` (required), the fixed I/O contract (`imgsz 640`, `[1,3,640,640]→
  [1,300,6]`). Only the swept axis moves.
- **Response variable**: `mAP50-95` (accuracy) — latency is a near-constant across QAT runs (same Q/DQ
  graph → ~1.4 ms kernel), so **accuracy is what the sweep resolves**. Latency is still measured on
  winners to confirm no regression.

### Center config (V6)
```text
LR0 1e-3 · LRF 0.01 · EPOCHS 50 / PATIENCE 10 · BATCH 16 · N_CALIB 128 · CALIB_SEED 42
quant INT8_DEFAULT_CFG (per-channel weights axis=0, per-tensor act, max calibrator) · amp False
```

---

## 2. Comparison protocol (how every run is scored)

Two-tier, to stay efficient:
```text
TIER 1 (every run, fast):   Ultralytics val mAP50-95 per epoch  → best value = the RANKING response
                            (free; already computed each epoch; drives early-stop + within-sweep ranking)
TIER 2 (sweep winner only): (a) pycocotools full-val on the fake-quant PyTorch model  → comparable to
                                the locked references (0.7564 / 0.7748), AND
                            (b) export → build Q/DQ INT8 engine → deployed full-val mAP + kernel latency
```
Rationale: Ultralytics-val and pycocotools differ in absolute value, so Tier-1 is used for **relative
ranking within a sweep** (same metric throughout), and Tier-2 puts the **winner** on the **same
pycocotools scale** as the baseline for the absolute claim.

### Comparison targets (pycocotools, full val, conf 0.001)
```text
PTQ baseline (locked)   mAP50-95 0.7564   ← QAT must BEAT this to justify itself
QAT V6 center           0.7644            ← the point each sweep perturbs
FP16 ceiling            0.7748            ← lossless target (the gap to close)
```

---

## 3. The sweeps (ordered highest-impact first)

### Sweep 1 — `LR0` (learning rate) — **highest impact**
| | |
|---|---|
| Values | {1e-4, 3e-4, **1e-3**, 3e-3, 1e-2} |
| New runs | 4 (center 1e-3 reused) |
| Question | What LR recovers accuracy without destabilizing the already-converged baseline? |
| Monitor focus | weight-shift (‖ΔW‖ — high LR moves weights too far → forgetting), scale-drift, loss stability, mAP-vs-epoch |
| Hypothesis | U-shape: too low → no adaptation (≈PTQ); too high → catastrophic forgetting (<PTQ). Optimum near 1e-3. |

### Sweep 2 — `EPOCHS` (fine-tune duration)
| | |
|---|---|
| Values | fixed-epoch runs {5, 10, 20, 35} (no early-stop, to trace the full accuracy-vs-duration curve) |
| New runs | 4 |
| Question | How long to fine-tune? (modelopt says ~10% = 5; V5's 10 was undertrained; V6 plateaued ~25) |
| Monitor | mAP-vs-epoch (where it plateaus), scale-drift stabilization, weight-shift saturation |
| Hypothesis | Rising then flat; plateau ~20–25; 5 ep insufficient for this task. |

### Sweep 3 — calibration method (`calibrator`)
| | |
|---|---|
| Values | {**max**, histogram/percentile-99.9, entropy/MSE} |
| New runs | 2 |
| Question | Do better initial scales (histogram/entropy vs max) close the gap to FP16? (docs/06 open item) |
| Monitor | epoch-0 per-layer SQNR (initial scale quality), final mAP |
| Hypothesis | Histogram/percentile may reduce outlier-driven clipping in attention-adjacent layers → small gain. |

### Sweep 4 — per-channel vs per-tensor **weights** (granularity ablation)
| | |
|---|---|
| Values | weights `axis=0` (**per-channel**) vs `axis=None` (per-tensor) |
| New runs | 1 |
| Question | How much does per-channel weight quant actually buy on this model? |
| Monitor | per-layer weight SQNR (per-tensor should drop, esp. high-variance filters), final mAP |
| Hypothesis | Per-tensor noticeably worse — quantifies the per-channel benefit (validates the default). |

### Sweep 5 — `N_CALIB` (calibration set size)
| | |
|---|---|
| Values | {32, **128**, 512} |
| New runs | 2 |
| Question | Does more calibration data give better frozen scales → better final accuracy? |
| Monitor | epoch-0 SQNR vs N_CALIB, final mAP |
| Hypothesis | Small effect (QAT adapts weights around frozen scales), but 32 may be too few. |

### Sweep 6 — `BATCH` size
| | |
|---|---|
| Values | {8, **16**, 32} |
| New runs | 2 |
| Question | Batch effect on QAT convergence/accuracy (gradient noise + BN running stats) |
| Monitor | training stability, weight-shift, final mAP |
| Hypothesis | Mild; larger batch smoother but may need LR rescale (kept fixed here → note the confound). |

> **Deferred (not on A100):** FP8 / INT4 formats — A100 has no FP8 tensor cores (Hopper-only), and
> INT4 needs block quant + heavy QAT. Out of scope for this hardware; revisit on the deployment GPU.

---

## 4. Per-run procedure (identical every run — reproducible)
```text
1. write run_config.json (center + the one swept value + seeds)
2. qat_run.sh  with the swept env var, monitoring callbacks attached
3. Tier-1: record best Ultralytics val mAP50-95 + all monitoring (scales/weights/SQNR/curves)
4. (winner of each sweep) Tier-2: pycocotools full-val on fake-quant model
                                  + export_qat_onnx → build_tensorrt_int8_qdq → engine mAP + latency
5. plot_run.py → per-run report;  after each sweep → summary.md overlay
```
Everything lands in `experiments/qat_iteration_2/sweeps/<run_name>/`.

---

## 5. Run count + wall-clock estimate

```text
Sweep                 new runs
1 LR0                    4
2 EPOCHS                 4
3 calibrator             2
4 per-ch/per-tensor      1
5 N_CALIB                2
6 BATCH                  2
center (V6) reused       1
------------------------------
TOTAL QAT runs          16
Tier-2 engine builds   ~6  (one winner per sweep)
```

Wall-clock (A100, YOLO26n QAT: fake-quant + amp=False + WORKERS=0 ≈ 5–8 min/epoch):
```text
~35-epoch run  ≈ 3–4 h      short 5-epoch run ≈ 0.5 h
weighted total ≈ ~40 GPU-hours training  +  ~2 h for the 6 engine builds/measures
```
On the **shared cluster** (GPUs intermittently free), realistic wall-clock is **several days**, one
run at a time under `nohup`, opportunistically grabbing an idle GPU (same pattern as the latency
sweeps). Fully sequential and resumable — no run depends on another (OFAT).

---

## 6. Reproducibility & rigor guarantees
- Fixed seeds (torch + `CALIB_SEED=42`) → deterministic except the swept axis.
- `run_config.json` per run captures the exact hyperparameter set → any run re-creatable.
- OFAT → every result attributable to one variable.
- Two-tier scoring → fast ranking, but the headline number is always **pycocotools full-val** on the
  same scale as the locked **0.7564** baseline.
- Monitoring (PART 2) means each run explains *why* it moved, not just *that* it moved (weight-shift,
  SQNR, scale drift), so the sweep produces mechanism, not just a leaderboard.

---

## 7. What we'll be able to conclude
- The **LR0 × EPOCHS** response surface (the two dominant knobs) → the recommended QAT recipe.
- Whether **calibration method / N_CALIB** can close the residual gap to FP16 (0.7644 → 0.7748).
- A **quantified per-channel-weight benefit** (Sweep 4) — a clean, teachable ablation.
- Per-layer **SQNR maps** identifying exactly which layers cap accuracy (expected: attention/head) —
  informing any future mixed-precision or selective-QAT decision.

**Proposed first action after sign-off:** Sweep 1 (`LR0`), since it's the highest-impact axis and its
result sets the sensible center for the rest.

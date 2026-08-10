# What Gets Monitored — Per Sweep (review, run nothing)

For each of the 6 planned sweeps: the signals most relevant to *that* parameter, what each tells us,
which layers to watch, and what a **good vs bad** run looks like in the plots. Signal definitions are
in `02_monitoring_scheme.md`. Reminder of the responses:
```text
mAP-vs-epoch        = Ultralytics val mAP50-95 each epoch (ranking)
‖ΔW‖ (weight-shift) = per-layer rel L2 of (W_epoch − W_epoch0)  — how far QAT moved a layer
SQNR (dB)           = per-layer signal-to-quant-noise — how much quantization perturbs a layer
amax / scale        = the frozen per-quantizer INT8 grid (set at calibration)
```

---

## Sweep 1 — `LR0` (learning rate) — the dominant knob

**Watch primarily:** `‖ΔW‖` per layer, mAP-vs-epoch, loss stability.
| Signal | What it tells us about LR |
|---|---|
| **‖ΔW‖ per layer** (final + per-epoch) | LR sets how far weights move. This is *the* LR signal — it shows whether we gently adapted or destroyed pretrained features. |
| **mAP-vs-epoch** | the outcome — did we beat PTQ (0.7564) and climb toward FP16, or crash? |
| **loss curve smoothness** | high LR → spiky/diverging loss; low LR → flat, barely moving |

**Layers watched most:** the **backbone early convs (model.0–9)**. They hold the pretrained
feature extractor; if LR is too high they shift hard and the model forgets what it learned (the
features the whole detector rests on). The **head** always moves most (it's re-fitting the quant grid)
— that's expected and fine; backbone movement is the danger sign.

**Good run:** `‖ΔW‖` **small in the backbone, larger in the head/late layers**; smooth decreasing
loss; mAP rises **above 0.7564** and plateaus near V6/FP16.
**Bad run:** *too high* → large uniform `‖ΔW‖` across the backbone + mAP **collapses below PTQ**
(catastrophic forgetting), spiky loss. *Too low* → `‖ΔW‖ ≈ 0` everywhere + mAP **stuck at ~PTQ** (no
recovery — QAT did nothing).
**The one plot:** per-layer `‖ΔW‖` bar (backbone should stay short) + mAP-vs-epoch overlaid across LRs.

---

## Sweep 2 — `EPOCHS` / patience (duration)

**Watch primarily:** mAP-vs-epoch (plateau point), `‖ΔW‖` saturation, scale/loss stabilization.
| Signal | What it tells us about duration |
|---|---|
| **mAP-vs-epoch** | where accuracy plateaus = the sufficient epoch count |
| **‖ΔW‖ saturation** | when per-layer shift stops growing = weights have finished adapting |
| **train-loss vs val-mAP divergence** | late-run overfitting (loss ↓ but val-mAP ↓) |

**Layers watched most:** **all**, but ordered by adaptation speed — **head/late layers plateau first**,
**backbone convs last**. The run is "done" when the **backbone `‖ΔW‖` flattens** and mAP stops rising.

**Good run:** mAP rises then **clearly flattens** for several epochs (you can read off "N epochs is
enough"); `‖ΔW‖` saturates; no val-mAP decline.
**Bad run:** mAP **still climbing at the cutoff** (undertrained — the V5=10-epoch failure mode), or
val-mAP **turns down late** while train-loss keeps dropping (overfitting → patience should have stopped
it).
**The one plot:** mAP-vs-epoch with the plateau annotated + `‖ΔW‖`-saturation curve.

---

## Sweep 3 — calibration method (`max` / histogram / entropy)

**Watch primarily:** **epoch-0 `amax` per layer** and **epoch-0 SQNR** (this sweep is about the
*starting* scales), then final mAP.
| Signal | What it tells us about the calibrator |
|---|---|
| **epoch-0 amax distribution** | `max` = widest (outlier-driven); histogram/percentile = clips the tail (tighter); entropy = KL-optimal grid |
| **epoch-0 SQNR per layer** | which calibrator gives the best *initial* quantization fidelity, layer by layer |
| **final mAP** | whether a better start survives the (frozen-scale) fine-tune into better accuracy |

**Layers watched most:** the **attention-adjacent and head layers** (`model.10`, `model.22`,
`model.23`) — high dynamic range / outlier-prone activations, exactly where `max` over-widens the
scale and wastes INT8 levels. **Backbone convs are well-behaved**, so the calibrator barely matters
there — don't over-read them.

**Good run:** histogram/entropy shows **higher epoch-0 SQNR on the high-range layers** (tighter scale,
less wasted range) and **final mAP closer to FP16 (0.7748)**.
**Bad run:** `max` clips/over-widens → **low SQNR on attention layers**; or the alternative calibrator
**moves nothing** (SQNR ≈ same) → calibrator is not the lever here.
**The one plot:** epoch-0 amax + SQNR per layer, one color per calibrator; final-mAP bar.

---

## Sweep 4 — per-channel vs per-tensor **weights** (granularity ablation)

**Watch primarily:** **weight scale spread** and **per-layer weight SQNR** for both settings.
| Signal | What it tells us |
|---|---|
| **per-channel amax spread within each conv** | how much output filters differ in range — the *reason* per-channel helps. Wide spread → per-tensor will clip. |
| **weight SQNR per layer (per-ch vs per-tensor)** | the direct cost of collapsing to one scale per layer |
| **final mAP delta** | the accuracy price of per-tensor weights |

**Layers watched most:** **convs with high inter-channel weight variance** — typically deeper convs
and the head branches. Those are where a single per-tensor scale hurts most; uniform-range convs are
indifferent.

**Good run (validates default):** per-channel keeps **weight SQNR high everywhere**; the per-tensor
run shows **SQNR drop on the high-spread convs** and a **measurable final-mAP loss** → cleanly
quantifies the per-channel benefit.
**Bad/uninteresting:** **no SQNR or mAP difference** → per-channel isn't buying anything on this model
(surprising, would be worth noting).
**The one plot:** per-conv channel-amax spread (box/violin) + weight-SQNR (per-ch vs per-tensor) + mAP delta.

---

## Sweep 5 — `N_CALIB` (calibration set size)

**Watch primarily:** **epoch-0 amax convergence vs N** and epoch-0 SQNR, then final mAP.
| Signal | What it tells us |
|---|---|
| **amax vs N_CALIB (per layer)** | does the scale estimate stabilize? if 128≈512, 128 is enough; if 32 is far off, it's too few |
| **epoch-0 SQNR vs N** | whether more frames improve the *starting* fidelity |
| **final mAP vs N** | whether any of it survives into accuracy |

**Layers watched most:** **activation quantizers in high-variance regions** (attention/head) — they
need more samples to estimate their range; backbone activation ranges stabilize with few frames.

**Good run:** amax **flat from 128→512** and final mAP **flat across N** → 128 is sufficient (confirms
the default).
**Bad run:** **N=32 gives unstable amax** (far from 128/512) and **lower mAP** → too few frames; or a
clear monotonic mAP gain with N → we're under-calibrating.
**The one plot:** amax-vs-N convergence per layer + final-mAP-vs-N.

---

## Sweep 6 — `BATCH` size

**Watch primarily:** **loss/mAP stability**, `‖ΔW‖`, and BN behavior. (Note: LR is held fixed → a
known LR×batch confound; flagged, not corrected, in this OFAT.)
| Signal | What it tells us |
|---|---|
| **loss variance / mAP-vs-epoch smoothness** | small batch = noisier gradients (more regularization, less stable); large = smoother |
| **‖ΔW‖** | whether larger batch under-moves weights at the fixed LR (underfit) |
| **BN running-stat drift** | batch size changes batch-norm statistics quality |

**Layers watched most:** **BN layers throughout** (their running stats depend on batch size) and
overall training stability.

**Good run:** stable loss, mAP **comparable to the center (batch 16)** → batch is not a sensitive axis.
**Bad run:** **batch 8 noisy/unstable** (jumpy mAP), or **batch 32 underfits** at the fixed LR (small
`‖ΔW‖`, mAP below center) — the latter is the confound, meaning "batch 32 *with rescaled LR*" is the
real follow-up.
**The one plot:** mAP-vs-epoch + loss-variance across batch sizes.

---

## Quick reference — the primary signal per sweep
```text
LR0            → per-layer ‖ΔW‖ (backbone!) + mAP-vs-epoch      "did we adapt or forget?"
EPOCHS         → mAP plateau + ‖ΔW‖ saturation                 "how long is enough?"
calibrator     → epoch-0 amax + SQNR on attention/head         "better starting scales?"
per-ch/tensor  → weight scale-spread + weight SQNR             "how much does per-channel buy?"
N_CALIB        → amax convergence vs N (attention/head)        "enough calibration frames?"
BATCH          → loss stability + BN drift + ‖ΔW‖              "convergence quality (LR-confounded)"
```

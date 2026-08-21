# Stage 8: Frozen Baseline and Per-Layer QAT Sensitivity

Machine and environment: **malmo** (H100 NVL), `~/venvs/medtronic-qat-p311` to train,
`~/venvs/medtronic-trt` to build and measure. See
[`00_environment_and_access.md`](00_environment_and_access.md).

Data, provenance and the full recap: [`results/experiments/week8/`](../results/experiments/week8/).
This document is the method and the reasoning; that folder holds the tables and the
per-layer sidecars.

---

## The question

Stages 3 to 6 established a precision ladder on the fully-trained model: FP32, FP16,
INT8 PTQ, INT8 QAT. They answer "how much does INT8 cost overall". They do not answer
the question that matters for a mixed-precision deployment: **which layers carry that
cost, and does the choice of layer move latency?**

Answering it needs one variable at a time. So the model is deliberately weakened
first: backbone and neck frozen, only the head trained. Everything except the
quantised layer is then held fixed across all 24 runs.

## Step 1: the frozen baselines (V0 and V1)

`train_frozen_baseline.py` takes COCO `yolo26n.pt`, freezes `model.0`–`model.22`
(`freeze=23`) and trains only the head on the surgical data for a fixed 50 epochs.
Deployed FP16, that is **V0**.

`apply_qat_0iter.py` then applies the standard modelopt quantisation to V0 with
**zero** training iterations: `mtq.quantize` plus max-calibration on 128
episode-diverse frames, seed 42, and nothing else. It exists because Ultralytics
cannot express `epochs=0`. Deployed INT8, that is **V1**.

V1 is the sweep's zero point: what INT8 costs before any fine-tuning recovers it.
It is PTQ reached through the QAT code path.

```text
                              mAP50    mAP50-95   kernel latency (A100)
frozen baseline, float         0.756     0.546     --
V0  frozen baseline, FP16      0.7419    0.5313    1.253 ms
V1  frozen + 0-iter QAT, INT8  0.6906    0.4892    1.503 ms
```

Two things are already visible here:

**Freezing is expensive.** 0.546 against the fully-trained baseline's 0.782: the COCO
backbone is not adapted to surgical imagery, and ~0.24 mAP50-95 is the price. That is
intended. It buys a model where one layer can be varied cleanly.

**V1 is worse than V0 on both axes**: it loses 0.042 mAP50-95 *and* costs 0.25 ms more.
Explicit-Q/DQ INT8 adds kernels, and on a launch-bound model that is a net loss.
Quantisation without fine-tuning is not a free trade here.

## Step 2: the sweep

For each of the 24 top-level modules in turn: quantise **that layer only**, fine-tune
for up to 6 epochs (patience 3), export Q/DQ ONNX, build an INT8 TensorRT engine, then
measure accuracy and latency. Everything else stays frozen FP16.

Six of the 24 (`model.11`, `12`, `14`, `15`, `18`, `21`) are Concat or Upsample nodes
with no trainable parameters. They short-circuit to a `no_trainable_params` row rather
than being silently dropped, which is what makes the master table a complete 24-row
record instead of one with unexplained holes. **18 layers** carry real results.

Train and deploy were run as two separate passes, not interleaved. That is deliberate:
it puts all 18 latency measurements inside one short exclusive-GPU window instead of
spreading them across ~20 hours of a shared machine's varying load.

```text
build      TensorRT 10.16.1.11, explicit Q/DQ (no calibrator), OPT_LEVEL=3, 8 GB workspace
accuracy   full val, 6,449 images, conf=0.001, max_dets=300, pycocotools COCOeval bbox
latency    trtexec, 300 iterations, 100 warmup, --noDataTransfers --useSpinWait,
           with and without --useCudaGraph, batch=1, H100
```

`OPT_LEVEL=3` was measured on this box, not assumed: level 1 gives 0.762 ms and level
5 gives 0.686 ms, but level 5 costs 6x the build time. Level 3 lands at 0.735 ms and
is *faster to build* than level 1. Every layer builds at the same level, so the
layer-to-layer ranking is unaffected either way.

## Result 1: layer choice dominates accuracy

```text
best     model.8    0.6675    recovers 62.4% of the V1 -> FP32 gap
         model.6    0.6525
         model.13   0.6518
...
worst    model.23   0.5025    below V0
         model.0    0.4982    below V0
         model.20   0.4949    below V0
         model.22   0.4874    below V1
```

Mid and late backbone (`model.4`–`model.10`, plus `model.13`) carries essentially all
the recoverable accuracy. The stem (`model.0`) and the head (`model.20`, `22`, `23`)
buy nothing: those four sit **below the V0 FP16 line**, meaning quantising them costs
more than six epochs of training them back recovers. `model.22` is fractionally below
even the calibrate-only V1 zero point.

Every layer clears V1, so fine-tuning always beats not fine-tuning. Only the layers
above V0 clear FP16, which is the bar that actually matters for deployment.

## Result 2: layer choice does not move latency, and that is a property of the design

CUDA-graph latency across all 18 engines spans **0.7396 to 0.7656 ms: a 3.5% spread**,
with no significant relationship to which layer was quantised.

This is easy to misread as "quantisation is latency-neutral". It is not. It is a
consequence of quantising exactly one layer:

- **No single layer is big enough.** The largest in the sweep is 9.1% of the model's
  2.59 G conv MACs, so even a perfect 2x speedup on it caps at ~4.5%. The observed
  3.5% spread is the same order as that ceiling.
- **A lone INT8 layer pays boundary costs.** It needs a quantize on input and a
  dequantize on output. At this size those reformats cost about what the INT8 kernel
  saves, which is why INT8 pays off in *contiguous runs*, not islands.
- **Launch overhead dominates.** CUDA graphs alone are worth **224 µs (23%)** on these
  same engines, roughly ten times anything layer choice achieves. At batch=1 this
  model is ~237 kernels in 0.75 ms, about 3.2 µs each: nowhere near saturating an H100.

Correlating CUDA-graph latency against candidate explanations (n=18, two-tailed 5%
critical |r| = 0.468):

```text
quantised layer's conv MACs      r = +0.182    not significant
quantised layer's param count    r = +0.274    not significant
layer index (stem -> head)       r = -0.250    not significant
engine kernel_count              r = -0.809    SIGNIFICANT
mAP50-95                         r = +0.273    not significant
```

Note the sign on MACs and params: quantising a *larger* layer trended slightly
**slower**, consistent with the reformat-boundary explanation. The only significant
correlate is `kernel_count`, i.e. how TensorRT happened to compile that graph rather
than how much arithmetic it contains. Attributing that properly needs per-engine
`trtexec --dumpProfile`, which has not been run.

## What this does and does not license

**It is a ranking experiment, not a deployment candidate.** The best single layer is
still 0.107 mAP50-95 short of the FP32 baseline.

**The obvious next step inverts it.** For a mixed-precision engine you want most
layers INT8 and the *sensitive* ones held in FP16: the sweep names those as the stem
and head. But two assumptions have to be made explicit first.

1. Ranking by "quantise L alone" and deciding by "quantise all but L" assumes
   quantisation errors are roughly additive across layers. A standard first-order
   approximation, adequate for ranking, not guaranteed.
2. **The ranking is measured on the frozen model** (mAP 0.49–0.67), not the
   fully-trained model where the PTQ and QAT ladder lives (0.75–0.77). Whether it
   transfers across that gap is untested.

The experiment that settles both is cheaper than this sweep was: on the *trained*
model, quantise everything except layer L with **no training at all**, then build and
evaluate. That measures the decision variable directly.

**And check what latency prize is actually on offer.** On the same A100 harness, FP32
2.019 ms, FP16 1.137 ms, INT8 PTQ 1.091 ms, INT8 QAT V6 1.397 ms. FP16 is both more
accurate *and* faster than INT8 QAT, so V6 is not on the Pareto frontier at all, and
PTQ buys 4% latency for -0.0177 mAP. The whole ladder spans 4%. On this model at
batch=1 the case for mixed precision is an accuracy case, not a speed case; the lever
for speed is batch size, which moves the model into the compute-bound regime where
tensor cores matter.

## Caveats

1. **Training budget was not perfectly uniform.** Early stopping cut some layers to
   4–5 epochs (`model.2` to 4; `model.5`, `7`, `23`, `0` to 5). `best_epoch` is
   0-indexed, so `model.2`'s best was its first epoch and it degraded after: a
   short early-peaking run, not a starved one. Small, but it does slightly confound
   layer-versus-layer comparison.
2. **Sweep latency and ladder latency are not on one axis.** The 18 engines were timed
   on an H100 with CUDA graphs; the ladder on an A100 with none. Figure 2 of the
   notebook keeps them in separate panels for exactly this reason.
3. **Engines have been cleared** to reclaim disk. Every sha256 and full build
   configuration survives in the committed `*.provenance.json` sidecars, and each is
   rebuildable from the retained `best_qat.onnx`.

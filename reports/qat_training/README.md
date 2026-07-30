# QAT Training — status & results (V5 fixed-10 vs V6 early-stopping)

Two QAT fine-tune runs of the V1 baseline (`best.pt`), INT8 fake-quant (modelopt
`INT8_DEFAULT_CFG`, 608 quantisers / 253 with scales). All metrics below are the
**Ultralytics validation mAP on the fake-quant model** (full val split, conf 0.001)
recorded per epoch — NOT the deployed engine (engine numbers are pycocotools; see caveat).

Plot: `qat_v5_v6_accuracy.png`.

## Parameters (identical except the epoch/patience regime)

```text
model best.pt · INT8_DEFAULT_CFG · lr0 1e-3 · lrf 0.01 · warmup 3 · amp False
batch 16 · imgsz 640 · workers 0 · single A100 (Geneva GPU 1) · seed default
V5:  EPOCHS=10  (fixed)                 -> ran all 10
V6:  EPOCHS=50 ceiling, PATIENCE=10     -> early-stopped at ep35 (best ep25)
```

## Runtime

```text
V5:  1 h 41 min   (10 epochs, ~10 min/epoch)
V6:  6 h 41 min   (35 epochs, ~11.5 min/epoch)
```

## Fluctuation (mAP50-95)

```text
              epochs   best        min     max     mean    std      range
V5 fixed-10     10    0.7427@ep10  0.6614  0.7427  0.7091  0.0256   0.0813
V6 early-stop   35    0.7593@ep25  0.6096  0.7593  0.7107  0.0346   0.1496
```

V6 fluctuates MORE (std 0.035 vs 0.026) — expected: its LR schedule is stretched over
50 epochs, so the LR stays higher for longer (more exploration/noise early) before
annealing. The noise shrinks in the back half as LR decays; the peak lands at ep25.

## Per-epoch mAP50-95

```text
ep   V5        V6            ep   V6
 1  0.6836    0.6962         19  0.7402
 2  0.6614    0.6824         20  0.7416
 3  0.6838    0.7094         21  0.7255
 4  0.7124    0.6469         22  0.7440
 5  0.7068    0.6967         23  0.7356
 6  0.7231    0.6484         24  0.6764
 7  0.7271    0.7147         25  0.7593  <- V6 BEST
 8  0.7338    0.6527         26  0.7223
 9  0.7161    0.6096         27  0.7343
10  0.7427*   0.6733         28  0.7165   (*V5 best, still climbing)
11   —        0.7367         29  0.7120
12   —        0.7125         30  0.7160
13   —        0.6678         31  0.7437
14   —        0.7144         32  0.7464
15   —        0.7051         33  0.7371
16   —        0.7385         34  0.7303
17   —        0.7408         35  0.7084  <- stop (10 ep after best, patience fired)
18   —        0.7380
```

## Result vs the precision bars (deployable pycocotools, for reference)

```text
FP32 engine   0.7747
FP16 engine   0.7748     <- accuracy leader
V6 QAT INT8   0.7593*    <- beats PTQ, ~0.015 below FP16   (*fake-quant metric; engine pending)
V4 PTQ INT8   0.7571     <- the bar V6 cleared (+0.0022)
V5 QAT INT8   0.7427     <- undertrained (fixed 10 ep, still climbing)
```

## Key takeaways

- **~25 epochs is the real convergence point.** V5's fixed 10 was undertrained (still
  climbing at ep10). V6's early stopping found the plateau at ep25 and confirmed it over
  10 patience epochs, then stopped.
- **V6 recovered accuracy past PTQ** (0.7593 > 0.7571) — the win QAT is meant to deliver —
  but stays below FP16/FP32, as expected for 8-bit vs 16-bit precision.
- **Metric caveat:** 0.7593 is the Ultralytics fake-quant val metric. The deployed INT8
  engine (pycocotools) is pending export+measure; on V5 the two matched within 0.001
  (0.7427 fake-quant vs 0.7437 engine), so V6's engine is expected ~0.759.
- **Latency is unchanged from V5** (~1.429 ms kernel) — graph-determined, weights don't
  affect it.

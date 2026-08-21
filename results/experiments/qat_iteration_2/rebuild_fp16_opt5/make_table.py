#!/usr/bin/env python3
"""Build the before/after table for the FP16+opt5 rebuild campaign.

before = original sweep (int8-only engine): mAP + kernel latency from sweeps/<run>/
after  = rebuilt engine (FP16+opt5): latency from rebuild_latency.json,
         accuracy from rebuild_fp16_opt5/<run>.engine.map_full.json (pass 2)
"""
import json, os
from pathlib import Path

REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
SWEEPS = REPO / "results/experiments/qat_iteration_2/sweeps"
OUT = REPO / "results/experiments/qat_iteration_2/rebuild_fp16_opt5"

RUNS = [("V2_lr0_1e-4","lr0=1e-4"),("V2_lr0_3e-4","lr0=3e-4"),("V2_lr0_3e-3","lr0=3e-3"),
        ("V2_lr0_1e-2","lr0=1e-2"),("V2_lrf_0.001","lrf=0.001"),("V2_lrf_0.1","lrf=0.1"),
        ("V2_batch_8","batch=8"),("V2_batch_32","batch=32"),("V2_ncalib_32","n_calib=32"),
        ("V2_ncalib_512","n_calib=512"),("V2_disable_attention","disable attn")]

lat = json.load(open(OUT / "rebuild_latency.json")) if (OUT/"rebuild_latency.json").exists() else {}


def loadj(p):
    try: return json.load(open(p))
    except Exception: return {}


def old_acc(r):
    for f in [f"{r}/best_qat_int8.engine.map_full.json", f"{r}/best_qat_int8.engine.map.json"]:
        d = loadj(SWEEPS / f)
        if d.get("map50_95") is not None: return d["map50_95"]
    return None


def old_lat(r):
    return loadj(SWEEPS / r / "latency.json").get("kernel_median_ms")


def new_acc(r):
    return loadj(OUT / f"{r}.engine.map_full.json").get("map50_95")


def new_lat(r):
    return (lat.get(r, {}).get("latency") or {}).get("median_ms")


def f(x, d=4):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "—"


rows = []
for r, knob in RUNS:
    oa, na = old_acc(r), new_acc(r)
    ol, nl = old_lat(r), new_lat(r)
    dl = f"{nl-ol:+.3f}" if isinstance(nl,(int,float)) and isinstance(ol,(int,float)) else "—"
    da = f"{na-oa:+.4f}" if isinstance(na,(int,float)) and isinstance(oa,(int,float)) else "—"
    st = lat.get(r, {})
    rows.append(f"| {r} | {knob} | {f(oa)} | {f(na)} | {da} | {f(ol,3)} | {f(nl,3)} | {dl} | "
                f"{st.get('engine_layers','—')} | {st.get('reformats','—')} |")

md = f"""# FP16+opt5 rebuild — before/after (no retraining)

Every trained QAT sweep model rebuilt on the improved TensorRT config
(INT8 + FP16 + opt-level 5). Same `best_qat.onnx`, same weights/scales — only the
engine build changed. Latency: clean CUDA-event median, exclusive GPU2, 300 iters.
Accuracy: full-val (6449 img) pycocotools, conf 0.001.

PTQ INT8 reference (floor): ~1.082 ms @ mAP50-95 0.7564.

| run | knob | old mAP | new mAP | Δacc | old ms | new ms | Δlat | layers | reformats |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
{chr(10).join(rows)}
"""
(OUT / "BEFORE_AFTER.md").write_text(md)
print(md)
print(f"wrote {OUT/'BEFORE_AFTER.md'}")

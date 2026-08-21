#!/usr/bin/env python3
"""Rebuild EVERY trained QAT sweep model on the improved TensorRT config
(INT8 + FP16 + opt-level 5 + DETAILED), and re-measure clean latency.

No retraining, no re-export: reuses each run's existing best_qat.onnx (the trained
Q/DQ model). Only the TensorRT engine build changes. Writes new engines here and a
latency/structure table; accuracy is measured in a second pass (accuracy is a
property of the ONNX, but FP16 fallback can shift it slightly, so we verify it).

Timing runs on an EXCLUSIVE idle GPU (set via CUDA_VISIBLE_DEVICES) — 300 iters
after 50 warmup, CUDA-event compute-only. Serial, so every number is comparable.
"""
import os, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
from pathlib import Path
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
SWEEPS = REPO / "results/experiments/qat_iteration_2/sweeps"
OUT = REPO / "results/experiments/qat_iteration_2/rebuild_fp16_opt5"
OUT.mkdir(parents=True, exist_ok=True)
LOG = trt.Logger(trt.Logger.ERROR)

RUNS = ["V2_lr0_1e-4","V2_lr0_3e-4","V2_lr0_3e-3","V2_lr0_1e-2","V2_lrf_0.001","V2_lrf_0.1",
        "V2_batch_8","V2_batch_32","V2_ncalib_32","V2_ncalib_512","V2_disable_attention"]


def CHK(r):
    err = r[0] if isinstance(r, (tuple, list)) else r
    if int(err) != 0:
        raise RuntimeError(f"CUDA err {err}")
    if isinstance(r, (tuple, list)):
        return r[1] if len(r) == 2 else r[1:]


def build(onnx_path, engine_path):
    builder = trt.Builder(LOG)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, LOG)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("parse failed")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.set_flag(trt.BuilderFlag.FP16)              # cheaper fallback for non-INT8 layers
    cfg.builder_optimization_level = 5              # max fusion search
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError("build returned None")
    Path(engine_path).write_bytes(bytes(ser))
    return net.num_layers, len(bytes(ser))


def bucket(dts):
    s = " ".join(dts)
    if "Int8" in s: return "INT8"
    if "Half" in s or "Float16" in s: return "FP16"
    if "Int32" in s: return "INT32"
    if "Float" in s: return "FP32"
    return "other"


def analyse(engine):
    parsed = json.loads(engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON))
    layers = parsed["Layers"] if isinstance(parsed, dict) else parsed
    prec, reformats = {}, 0
    for L in layers:
        if not isinstance(L, dict): continue
        if "Reformat" in L.get("LayerType", ""): reformats += 1
        p = bucket(sorted({o.get("Format/Datatype", "?") for o in (L.get("Outputs") or [])}))
        prec[p] = prec.get(p, 0) + 1
    return len(layers), reformats, prec


def timed(engine, iters=300, warmup=50):
    ctx = engine.create_execution_context()
    bufs = {}
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        shp = [max(1, int(x)) for x in engine.get_tensor_shape(nm)]
        d = CHK(cudart.cudaMalloc(int(np.prod(shp)) * 4)); bufs[nm] = d
        ctx.set_tensor_address(nm, int(d))
    stream = CHK(cudart.cudaStreamCreate())
    s = CHK(cudart.cudaEventCreate()); e = CHK(cudart.cudaEventCreate())
    for _ in range(warmup): ctx.execute_async_v3(int(stream))
    CHK(cudart.cudaStreamSynchronize(stream))
    ts = []
    for _ in range(iters):
        CHK(cudart.cudaEventRecord(s, stream)); ctx.execute_async_v3(int(stream))
        CHK(cudart.cudaEventRecord(e, stream)); CHK(cudart.cudaStreamSynchronize(stream))
        ts.append(CHK(cudart.cudaEventElapsedTime(s, e)))
    for d in bufs.values(): CHK(cudart.cudaFree(d))
    ts = np.array(ts)
    return {"median_ms": float(np.median(ts)), "mean_ms": float(ts.mean()),
            "min_ms": float(ts.min()), "max_ms": float(ts.max())}


def main():
    CHK(cudart.cudaSetDevice(0))
    runtime = trt.Runtime(LOG)
    results = {}
    for r in RUNS:
        onnx = SWEEPS / r / "best_qat.onnx"
        if not onnx.exists():
            print(f"SKIP {r}: no onnx"); continue
        eng = OUT / f"{r}.engine"
        print(f"\n=== {r} ===")
        net_layers, nbytes = build(onnx, eng)
        e = runtime.deserialize_cuda_engine(eng.read_bytes())
        n_layers, reformats, prec = analyse(e)
        t = timed(e)
        results[r] = {"engine_layers": n_layers, "reformats": reformats, "precision": prec,
                      "size_mb": round(nbytes/1e6, 2), "latency": t}
        print(f"  layers={n_layers} reformats={reformats} prec={prec} "
              f"median={t['median_ms']:.4f} ms")
        del e
    (OUT / "rebuild_latency.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT/'rebuild_latency.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-layer verification of the production TensorRT engines.

trtexec is NOT installed in this environment (pip `tensorrt` wheel ships the
Python API but no binary). This script reproduces the exact data the requested
trtexec flags would have exported, via the TensorRT Python API:

  --loadEngine                -> runtime.deserialize_cuda_engine
  --exportLayerInfo           -> EngineInspector.get_engine_information(JSON)   [static]
  --exportProfile / --dumpProfile -> IProfiler on the execution context        [instrumented run]
  --separateProfileRun        -> a SEPARATE clean CUDA-event timing run (no profiler)

Structure (precision, layer types, reformats) is static and trustworthy on any
GPU. Timings require an EXCLUSIVE idle GPU -- pass one via DEVICE.

Precision inference: there is NO "Precision" field on an inspector layer. The
precision a layer ran at is carried by its OUTPUT tensors' Format/Datatype
(same logic build_tensorrt_engine.py already relies on).
"""
import os, re, json, sys
from pathlib import Path
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
OUTROOT = REPO / "experiments/qat_iteration_2/engine_verification"
DEVICE = int(os.environ.get("DEVICE", "-1"))
TIMED_ITERS = int(os.environ.get("TIMED_ITERS", "300"))
WARMUP = int(os.environ.get("WARMUP", "50"))
PROFILE_ITERS = int(os.environ.get("PROFILE_ITERS", "100"))
DO_TIMING = DEVICE >= 0

ENGINES = [
    ("FP32",          "models/yolo26n_sanoscience_full_left/baseline/fp32/best_fp32.engine"),
    ("FP16",          "models/yolo26n_sanoscience_full_left/baseline/fp16/best_fp16.engine"),
    ("V4_int8_ptq",   "models/yolo26n_sanoscience_full_left/baseline/int8_ptq/best_int8.engine"),
    ("PTQ_int8_fp16", "experiments/qat_iteration_2/ptq_baseline/best_int8_fp16.engine"),
    ("PTQ_maxfp16",   "experiments/qat_iteration_2/ptq_baseline/best_int8_fp16_max.engine"),
    ("V6_qat",        "models/yolo26n_sanoscience_full_left/qat/v6_final/best_qat_int8.engine"),
]

LOGGER = trt.Logger(trt.Logger.ERROR)


def CHK(ret):
    err = ret[0] if isinstance(ret, (tuple, list)) else ret
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    if isinstance(ret, (tuple, list)):
        return ret[1] if len(ret) == 2 else ret[1:]
    return None


DTYPE_BYTES = {trt.DataType.FLOAT: 4, trt.DataType.HALF: 2, trt.DataType.INT8: 1,
               trt.DataType.INT32: 4, trt.DataType.BOOL: 1}
try:
    DTYPE_BYTES[trt.DataType.INT64] = 8
except AttributeError:
    pass


def bucket_precision(datatypes):
    """Map a set of Format/Datatype strings -> a single precision bucket."""
    s = " ".join(datatypes)
    if "Int8" in s:
        return "INT8"
    if "Half" in s or "FP16" in s or "Float16" in s:
        return "FP16"
    if "Int32" in s:
        return "INT32"
    if "Float" in s or "FP32" in s:
        return "FP32"
    return "other"


MODEL_RE = re.compile(r"model\.(\d+)")


def module_index(name):
    m = MODEL_RE.search(name or "")
    return int(m.group(1)) if m else None


class LayerProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.times = {}   # name -> summed ms

    def report_layer_time(self, layer_name, ms):
        self.times[layer_name] = self.times.get(layer_name, 0.0) + ms


def analyse_structure(engine):
    """Static inspector parse -> (raw_layers, per-layer records, summary)."""
    insp = engine.create_engine_inspector()
    raw = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    parsed = json.loads(raw)
    layers = parsed["Layers"] if isinstance(parsed, dict) else parsed

    records, prec_count, type_count = [], {}, {}
    reformats = []
    detailed = False
    for L in layers:
        if not isinstance(L, dict):
            # verbosity was not DETAILED -> layers come back as bare strings
            records.append({"name": str(L), "type": "?", "precision": "?", "out_dtypes": []})
            continue
        detailed = True
        name = L.get("Name", "")
        ltype = L.get("LayerType", "?")
        out_dts = sorted({o.get("Format/Datatype", "?") for o in (L.get("Outputs") or [])})
        in_dts = sorted({o.get("Format/Datatype", "?") for o in (L.get("Inputs") or [])})
        prec = bucket_precision(out_dts)
        prec_count[prec] = prec_count.get(prec, 0) + 1
        type_count[ltype] = type_count.get(ltype, 0) + 1
        rec = {"name": name, "type": ltype, "precision": prec,
               "out_dtypes": out_dts, "in_dtypes": in_dts,
               "module_index": module_index(name)}
        records.append(rec)
        is_reformat = ("Reformat" in ltype) or ("reformat" in name.lower()) \
                      or ("copy" in name.lower() and ltype == "?")
        if is_reformat:
            reformats.append(rec)
    summary = {"n_layers": len(layers), "detailed_verbosity": detailed,
               "precision_count": prec_count, "type_count": type_count,
               "n_reformat": len(reformats)}
    return raw, records, reformats, summary


def io_tensors(engine):
    ios = []
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        ios.append({
            "name": nm,
            "mode": str(engine.get_tensor_mode(nm)).split(".")[-1],
            "shape": [int(x) for x in engine.get_tensor_shape(nm)],
            "dtype": engine.get_tensor_dtype(nm),
        })
    return ios


def alloc_io(engine, ctx):
    ios = io_tensors(engine)
    bufs = {}
    for t in ios:
        n = int(np.prod([max(1, s) for s in t["shape"]]))
        nbytes = n * DTYPE_BYTES.get(t["dtype"], 4)
        d = CHK(cudart.cudaMalloc(nbytes))
        bufs[t["name"]] = (d, nbytes)
        ctx.set_tensor_address(t["name"], int(d))
    return ios, bufs


def timed_run(engine):
    """Clean CUDA-event compute-only timing (the --separateProfileRun analogue)."""
    ctx = engine.create_execution_context()
    ios, bufs = alloc_io(engine, ctx)
    stream = CHK(cudart.cudaStreamCreate())
    # seed input buffers with random data
    inp = next(t for t in ios if t["mode"] == "INPUT")
    n = int(np.prod([max(1, s) for s in inp["shape"]]))
    host = (np.random.rand(n).astype(np.float32)
            if inp["dtype"] == trt.DataType.FLOAT
            else np.random.rand(n).astype(np.float16))
    CHK(cudart.cudaMemcpyAsync(bufs[inp["name"]][0], host.ctypes.data, host.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    ev_s = CHK(cudart.cudaEventCreate())
    ev_e = CHK(cudart.cudaEventCreate())
    for _ in range(WARMUP):
        ctx.execute_async_v3(int(stream))
    CHK(cudart.cudaStreamSynchronize(stream))
    times = []
    for _ in range(TIMED_ITERS):
        CHK(cudart.cudaEventRecord(ev_s, stream))
        ctx.execute_async_v3(int(stream))
        CHK(cudart.cudaEventRecord(ev_e, stream))
        CHK(cudart.cudaStreamSynchronize(stream))
        times.append(CHK(cudart.cudaEventElapsedTime(ev_s, ev_e)))
    for d, _ in bufs.values():
        CHK(cudart.cudaFree(d))
    CHK(cudart.cudaStreamDestroy(stream))
    times = np.array(times)
    return {"median_ms": float(np.median(times)), "mean_ms": float(times.mean()),
            "min_ms": float(times.min()), "max_ms": float(times.max()),
            "std_ms": float(times.std()), "n": len(times)}


def profile_run(engine):
    """Instrumented per-layer profiling (the --dumpProfile analogue).

    ABSOLUTE times are inflated by instrumentation; use for per-layer SHARE, not
    absolute latency. That is exactly why trtexec keeps profiling separate from
    timing (--separateProfileRun)."""
    ctx = engine.create_execution_context()
    prof = LayerProfiler()
    ctx.profiler = prof
    ios, bufs = alloc_io(engine, ctx)
    stream = CHK(cudart.cudaStreamCreate())
    for _ in range(10):
        ctx.execute_async_v3(int(stream))
    CHK(cudart.cudaStreamSynchronize(stream))
    prof.times.clear()
    for _ in range(PROFILE_ITERS):
        ctx.execute_async_v3(int(stream))
    CHK(cudart.cudaStreamSynchronize(stream))
    for d, _ in bufs.values():
        CHK(cudart.cudaFree(d))
    CHK(cudart.cudaStreamDestroy(stream))
    per_layer = {k: v / PROFILE_ITERS for k, v in prof.times.items()}
    return per_layer


def main():
    if DO_TIMING:
        CHK(cudart.cudaSetDevice(DEVICE))
        print(f"[timing ON] DEVICE={DEVICE}  timed_iters={TIMED_ITERS} profile_iters={PROFILE_ITERS}")
    else:
        print("[timing OFF] structure-only (no DEVICE set)")

    runtime = trt.Runtime(LOGGER)
    index = {}
    for name, rel in ENGINES:
        path = REPO / rel
        outdir = OUTROOT / name
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name}  ({rel}) ===")
        eng = runtime.deserialize_cuda_engine(path.read_bytes())
        if eng is None:
            print("  DESERIALIZE FAILED"); index[name] = {"error": "deserialize failed"}; continue

        raw, records, reformats, summary = analyse_structure(eng)
        (outdir / "layer_info.json").write_text(raw)
        (outdir / "layers_parsed.json").write_text(json.dumps(records, indent=2))
        ios = io_tensors(eng)
        summary["io"] = [{"name": t["name"], "mode": t["mode"], "shape": t["shape"],
                          "dtype": str(t["dtype"]).split(".")[-1]} for t in ios]
        print(f"  layers={summary['n_layers']} detailed={summary['detailed_verbosity']} "
              f"prec={summary['precision_count']} reformats={summary['n_reformat']}")

        if DO_TIMING:
            try:
                summary["timing_clean"] = timed_run(eng)
                print(f"  clean compute median={summary['timing_clean']['median_ms']:.4f} ms")
            except Exception as e:
                summary["timing_clean"] = {"error": str(e)}
                print(f"  timing FAILED: {e}")
            try:
                pl = profile_run(eng)
                (outdir / "profile_per_layer.json").write_text(json.dumps(pl, indent=2))
                total = sum(pl.values())
                # attach instrumented time to records + aggregate reformat share
                by_name = {r["name"]: r for r in records}
                for nm, ms in pl.items():
                    if nm in by_name:
                        by_name[nm]["profile_ms"] = ms
                ref_ms = sum(ms for nm, ms in pl.items()
                             if nm in by_name and by_name[nm] in reformats)
                summary["profile_instrumented"] = {
                    "total_ms": total, "reformat_ms": ref_ms,
                    "reformat_share_pct": (100 * ref_ms / total) if total else None,
                    "n_layers_timed": len(pl)}
                (outdir / "layers_parsed.json").write_text(json.dumps(records, indent=2))
                print(f"  instrumented total={total:.4f} ms  reformat={ref_ms:.4f} ms "
                      f"({summary['profile_instrumented']['reformat_share_pct']:.1f}%)")
            except Exception as e:
                summary["profile_instrumented"] = {"error": str(e)}
                print(f"  profiling FAILED: {e}")

        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        index[name] = summary
        del eng

    (OUTROOT / "_index.json").write_text(json.dumps(index, indent=2))
    print(f"\nWrote {OUTROOT/'_index.json'}")


if __name__ == "__main__":
    main()

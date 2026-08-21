"""Per-layer metrics for the Week-8 layer sweep, MALMO / H100 NVL edition.

Differs from make_metrics.py (Geneva/A100) in one way that matters for reporting:
every record carries its hardware provenance and an explicit statement of how
exclusive the GPU was while latency was measured. The sweep moved boxes mid-run
(Geneva A100-SXM4-80GB -> Malmo H100 NVL 94GB), so a row's latency is only
interpretable next to the box it came from. Accuracy is hardware-independent and
stays directly comparable across both.
"""
import os, json, glob, subprocess
from pathlib import Path

L = int(os.environ["L"])
OUT = Path(os.environ["OUT"])
NG = os.environ.get("NG", "")            # trtexec median, no CUDA graph
G = os.environ.get("G", "")              # trtexec median, with CUDA graph
EPOCHS = int(os.environ.get("EPOCHS", 12))
PATIENCE = int(os.environ.get("PATIENCE", 4))
STATUS = os.environ.get("STATUS", "ok")
OPT_LEVEL = os.environ.get("OPT_LEVEL", "3")   # TensorRT builder search depth
DEVICE = os.environ.get("GPU", "1")
ENGINE = OUT / os.environ.get("ENGINE_NAME", "engine_int8_h100.engine")
REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
SWEEP = REPO / "experiments/week8_layer_sweep"

# The flag the whole H100 latency column has to be read through. Malmo GPU 1 is
# shared and other users' jobs come and go mid-sweep, so exclusivity is a fact
# about EACH layer's reading, not a property of the run. It is therefore DERIVED
# from the snapshot taken at metrics time (gpu_state, below). An earlier version
# hardcoded "resident process present (2.9GB)" -- one snapshot's truth, which
# would have been stamped onto every row no matter what was actually resident.
def exclusivity(state):
    if "error" in state:
        return f"unknown - could not read GPU state ({state['error']})"
    others = state.get("foreign_compute_procs", [])
    if not others:
        return (f"exclusive - no other compute process on GPU {state['index']} "
                f"at metrics time (util {state['util_pct']}%)")
    mem = sum(m for _, m in others)
    return (f"NOT exclusive - {len(others)} other compute process(es) on GPU "
            f"{state['index']} holding {mem} MiB at metrics time "
            f"(util {state['util_pct']}%)")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def gpu_state(idx):
    """Snapshot the device at metrics time, so the exclusivity flag is backed by
    a reading rather than by a claim made once at the top of the sweep.

    COMPUTE apps only. This box always has Xorg holding a graphics context, and
    counting graphics contexts reports an idle GPU as busy -- the same trap the
    benchmark_latency.py gate hit. Our own PID is separated out so that this
    script's TensorRT deserialize does not get reported as contention."""
    try:
        q = subprocess.run(
            ["nvidia-smi", "-i", str(idx), "--format=csv,noheader,nounits",
             "--query-gpu=name,utilization.gpu,memory.used,temperature.gpu"],
            capture_output=True, text=True, timeout=30).stdout.strip().split(", ")
        procs = subprocess.run(
            ["nvidia-smi", "-i", str(idx), "--format=csv,noheader,nounits",
             "--query-compute-apps=pid,used_memory"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        mine, parsed = os.getpid(), []
        for line in procs.splitlines():
            if not line.strip():
                continue
            pid, mem = [x.strip() for x in line.split(",")[:2]]
            parsed.append((int(pid), int(mem)))
        return {"index": int(idx), "gpu_name": q[0], "util_pct": int(q[1]),
                "mem_used_mb": int(q[2]), "temp_c": int(q[3]),
                "compute_procs": [f"{p}, {m}" for p, m in parsed],
                "foreign_compute_procs": [[p, m] for p, m in parsed if p != mine]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---- accuracy (full val, conf=0.001 -- see CLAUDE.md; never conf=0.25) -------
# STRICT: only an accuracy file produced by an engine built on THIS box counts.
# The previous fallback to "*.map_full.json" would silently adopt the Geneva/A100
# engine's mAP for layers 0/13/16/17 and stamp it with an H100 hardware record --
# a provenance lie, not a rounding error. No H100 accuracy file -> map is null.
acc, acc_src = {}, None
for p in sorted(glob.glob(str(OUT / "*_h100.engine.map_full.json"))):
    acc, acc_src = json.load(open(p)), Path(p).name
    break
map50, map5095 = acc.get("map50"), acc.get("map50_95")

# ---- engine size + kernel count --------------------------------------------
size_mb = round(ENGINE.stat().st_size / 1e6, 2) if ENGINE.exists() else None
kcount = devmem = trt_version = None
if ENGINE.exists():
    try:
        import tensorrt as trt
        trt_version = trt.__version__
        R = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        e = R.deserialize_cuda_engine(ENGINE.read_bytes())
        d = json.loads(e.create_engine_inspector().get_engine_information(
            trt.LayerInformationFormat.JSON))
        kcount = len(d["Layers"] if isinstance(d, dict) else d)
        try:
            devmem = round((e.device_memory_size_v2 if hasattr(e, "device_memory_size_v2")
                            else e.device_memory_size) / 1e6, 1)
        except Exception:
            devmem = None
    except Exception:
        pass

# ---- epochs actually trained / best epoch ----------------------------------
epochs_trained = best_epoch = None
rc = glob.glob(str(OUT / "train" / "**" / "results.csv"), recursive=True)
if rc:
    rows = open(rc[0]).read().strip().splitlines()
    epochs_trained = len(rows) - 1
# best epoch = the last "best fitness ... @ epoch N" the QAT callback printed.
# The TRAIN phase's log is the only one that carries it: run_layer_malmo.sh names
# its log run_malmo_$PHASE.log and truncates it at entry, so the deploy pass's own
# log holds no training output. Earlier this read "run_malmo.log", a name the
# script never writes, so best_epoch came out null for every layer in the sweep.
# run.log is the Geneva-era name, kept as a fallback for reused training.
import re
log = None
for _name in ("run_malmo_train.log", "run_malmo.log", "run.log"):
    _p = OUT / _name
    if _p.exists() and "best fitness" in _p.read_text(errors="ignore"):
        log = _p
        break
if log is not None:
    try:
        hits = re.findall(r"best fitness ([0-9.]+) @ epoch (\d+)",
                          log.read_text(errors="ignore"))
        if hits:
            best_epoch = int(hits[-1][1])
    except Exception:
        pass

# What per-epoch validation actually ran on, taken from the marker the train
# phase wrote (_trained_malmo.json -> "val_during_training").
try:
    val_during = json.load(open(OUT / "_trained_malmo.json"))["val_during_training"]
except Exception:
    val_during = "unknown"

_state = gpu_state(DEVICE)          # one reading, used for both fields below

ng, g = f(NG), f(G)
fps = round(1000.0 / g, 1) if g else (round(1000.0 / ng, 1) if ng else None)
V1 = 0.489                                   # 0-iteration QAT reference (mAP50-95)

metrics = dict(
    trained_layer=f"model.{L}", layer=L, status=STATUS,
    map50=map50, map50_95=map5095,
    delta_vs_V1=(round(map5095 - V1, 4) if map5095 else None),
    latency_no_graph_ms=ng, latency_graph_ms=g, fps=fps,
    kernel_count=kcount, engine_size_mb=size_mb, gpu_mem_mb=devmem,
    epochs_config=EPOCHS, patience_config=PATIENCE,
    builder_optimization_level=int(OPT_LEVEL),
    accuracy_source=acc_src,          # None => no H100 accuracy file was found
    # Read from the training marker, never asserted here. This field said
    # "val100_seed42_subset" -- a plan that was proposed and then REVERTED before
    # the sweep launched; every layer validated on the full 6,449-image set. A
    # constant would have written that false methodology onto all 24 rows.
    val_during_training=val_during,
    accuracy_eval_set="full_val_6449_conf0.001",
    epochs_trained=epochs_trained, best_epoch=best_epoch,
    hardware=dict(host="malmo.ee.ucl.ac.uk", gpu="NVIDIA H100 NVL 94GB",
                  device_index=int(DEVICE), tensorrt=trt_version,
                  previous_box="geneva / A100-SXM4-80GB",
                  # Where the QAT fine-tune itself ran. Weights are hardware-
                  # independent, so a Geneva-trained layer is not retrained --
                  # only its engine/accuracy/latency are redone here.
                  trained_on=os.environ.get("TRAINED_ON", "malmo / H100 NVL"),
                  engine_built_on="malmo / H100 NVL"),
    latency_exclusivity=exclusivity(_state),
    latency_comparable_to_a100_scorecard=False,
    accuracy_comparable_to_a100_scorecard=True,
    gpu_state_at_metrics=_state,
)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

# ---- master CSV (header always present; one row per layer, de-duplicated) ---
csv = SWEEP / "results_master_malmo_h100.csv"
hdr = ("layer,status,mAP50,mAP50-95,delta_vs_V1,latency_no_graph_ms,"
       "latency_graph_ms,fps,kernel_count,engine_size_mb,epochs_trained,best_epoch\n")
if not csv.exists():
    with open(csv, "w") as c:
        c.write(hdr)


def s(x):
    return "" if x is None else x


row = (f"model.{L},{STATUS},{s(map50)},{s(map5095)},{s(metrics['delta_vs_V1'])},"
       f"{s(ng)},{s(g)},{s(fps)},{s(kcount)},{s(size_mb)},{s(epochs_trained)},"
       f"{s(best_epoch)}\n")
lines = [l for l in open(csv).read().splitlines(keepends=True)
         if not l.startswith(f"model.{L},")]
if not lines or not lines[0].startswith("layer,"):
    lines.insert(0, hdr)
open(csv, "w").write("".join(lines) + row)
print(f"[metrics] layer {L}: status={STATUS} mAP50-95={map5095} "
      f"lat_ng={ng} lat_g={g} kernels={kcount} epochs={epochs_trained}")

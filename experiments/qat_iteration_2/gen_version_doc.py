import os, json
from pathlib import Path

# Generate version.md (status table + metrics) for one sweep version, and append a row to
# master_comparison.md. Reads the sidecars produced by the reused Phase-1 scripts.
OUT = Path(os.environ["OUT"])          # sweeps/<VERSION>/
VERSION = os.environ["VERSION"]
MASTER = OUT.parent / "master_comparison.md"

# V2_baseline reference (QAT V6 defaults) for deltas until V2_baseline is itself run.
BASE = {"map50": 0.9321, "map50_95": 0.7644, "kernel_ms": 1.39}


def loadj(name, default=None):
    p = OUT / name
    try:
        return json.load(open(p))
    except Exception:
        return default


cfg = loadj("run_config.json", {})
acc = loadj("best_qat_int8.engine.map_full.json") or loadj("best_qat_int8.engine.map.json") or {}
bp = loadj("best_qat_int8.engine.provenance.json", {})
op = loadj("best_qat.onnx.provenance.json", {})
lat = loadj("latency.json", {})

map50, map5095 = acc.get("map50"), acc.get("map50_95")
eval_n = acc.get("n_images")
eval_label = (("val100" if eval_n <= 200 else "full") + f" ({eval_n})") if eval_n else "—"
kernel = lat.get("kernel_median_ms")
kernel_disp = f"{kernel:.3f}" if isinstance(kernel, (int, float)) else "not measured (non-exclusive)"
qdq = op.get("quantize_linear_nodes")            # Q/DQ pairs = QAT engine coverage (always available)
size_mb = round(bp.get("engine_bytes", 0) / 1e6, 2) if bp.get("engine_bytes") else None

# ---- hyperparameter status table (all default except the tested knob) ----
DEFAULT = {"lr0": "1e-3", "lrf": "0.01", "epochs": "50", "patience": "10", "weight_axis": "0",
           "disable_layers": "", "calib_method": "max", "calib_percentile": "99.99",
           "n_calib": "128", "batch": "16"}
LABEL = {"lr0": "LR0", "lrf": "LRF", "epochs": "EPOCHS", "patience": "PATIENCE", "weight_axis": "WEIGHT_AXIS",
         "disable_layers": "DISABLE_LAYERS", "calib_method": "CALIB_METHOD",
         "calib_percentile": "CALIB_PERCENTILE", "n_calib": "N_CALIB", "batch": "BATCH"}
rows, active = [], []
for k in ["lr0", "lrf", "epochs", "patience", "weight_axis", "disable_layers",
          "calib_method", "calib_percentile", "n_calib", "batch"]:
    cur = str(cfg.get(k, DEFAULT[k]) or DEFAULT[k])
    dflt = DEFAULT[k]
    changed = cur != dflt
    show_cur = cur if cur else "(none)"
    show_dflt = dflt if dflt else "(none)"
    mark = "✅ CHANGED" if changed else "—"
    rows.append(f"| {LABEL[k]} | {show_cur} | {show_dflt} | {mark} |")
    if changed:
        active.append(f"{LABEL[k]}={show_cur}")

active_str = ", ".join(active) if active else "(none — baseline defaults = V6)"


def fmt(x, d=4):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "—"


def dlt(x, b):
    return f"{x-b:+.4f}" if isinstance(x, (int, float)) else "—"


md = f"""# {VERSION}

**Active knob(s):** {active_str} — everything else at baseline.

## Hyperparameter status (all default except the tested knob)
| knob | this run | baseline | changed? |
|---|---|---|---|
{chr(10).join(rows)}

## Metrics (same method as baseline: full-val pycocotools conf 0.001 · CUDA-event kernel)
| metric | this run | V2_baseline (=V6) | Δ |
|---|---|---|---|
| mAP50 | {fmt(map50)} | {BASE['map50']:.4f} | {dlt(map50, BASE['map50'])} |
| mAP50-95 | {fmt(map5095)} | {BASE['map50_95']:.4f} | {dlt(map5095, BASE['map50_95'])} |
| kernel ms | {kernel_disp} | {BASE['kernel_ms']:.3f} | {dlt(kernel, BASE['kernel_ms']) if isinstance(kernel,(int,float)) else '—'} |
| size MB | {size_mb or '—'} | — | — |
| Q/DQ pairs | {qdq or '—'} | 207 | — |
| eval set | {eval_label} | full (6449) | — |
"""
(OUT / "version.md").write_text(md)
print(f"[doc] wrote {OUT/'version.md'}")

# machine-readable metrics for sweep-level aggregation (gen_sweep_doc.py)
(OUT / "metrics.json").write_text(json.dumps({
    "version": VERSION, "knobs": cfg,
    "map50": map50, "map50_95": map5095, "kernel_ms": kernel,
    "size_mb": size_mb, "qdq_pairs": qdq, "eval_n": eval_n,
}, indent=2))

# ---- append to master comparison ----
header = ("| version | active knob | mAP50 | mAP50-95 | Δ50-95 | kernel ms | size MB | Q/DQ |\n"
          "|---|---|---|---|---|---|---|---|\n")
if not MASTER.exists():
    MASTER.write_text("# QAT Iteration 2 — master comparison\n\n"
                      "All versions vs V2_baseline (=V6 defaults). Same metric methods as the locked PTQ baseline.\n\n"
                      + header)
row = (f"| {VERSION} | {active_str} | {fmt(map50)} | {fmt(map5095)} | "
       f"{dlt(map5095, BASE['map50_95'])} | {fmt(kernel,3) if isinstance(kernel,(int,float)) else '—'} | "
       f"{size_mb or '—'} | {qdq or '—'} |\n")
# de-dup: drop any existing row for this version, then append
lines = MASTER.read_text().splitlines(keepends=True)
lines = [ln for ln in lines if not ln.startswith(f"| {VERSION} |")]
MASTER.write_text("".join(lines) + row)
print(f"[doc] appended row to {MASTER}")

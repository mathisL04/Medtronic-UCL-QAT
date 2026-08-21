import os, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Aggregate a NON-BINARY knob sweep (multiple version runs of one knob) into:
#   - a value -> metric table
#   - accuracy-vs-value plot (mAP50-95 + mAP50)
#   - latency-vs-value plot (kernel ms; flagged "invariant" if flat)
# into sweeps/<KNOB>_sweep/.  Each version must have been produced by run_version.sh
# (so it has metrics.json). Binary knobs do NOT use this — they keep the A-vs-B version docs.
#
# Usage:
#   KNOB=calib_percentile VERSIONS=V2_calib_pct_99.9,V2_calib_pct_99.99,V2_calib_pct_99.999 \
#     python results/experiments/qat_iteration_2/gen_sweep_doc.py
SWEEPS = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/results/experiments/qat_iteration_2/sweeps")
KNOB = os.environ["KNOB"]                          # run_config key that varies, e.g. "calib_percentile"
VERSIONS = [v.strip() for v in os.environ["VERSIONS"].split(",") if v.strip()]
OUTDIR = SWEEPS / f"{KNOB}_sweep"
OUTDIR.mkdir(parents=True, exist_ok=True)

rows = []
for v in VERSIONS:
    p = SWEEPS / v / "metrics.json"
    if not p.exists():
        print(f"  WARN: {p} missing — skipping {v}"); continue
    m = json.load(open(p))
    val = m.get("knobs", {}).get(KNOB)
    rows.append({"version": v, "value": val, "map50": m.get("map50"),
                 "map50_95": m.get("map50_95"), "kernel_ms": m.get("kernel_ms")})

# sort by numeric value where possible
def _key(r):
    try: return (0, float(r["value"]))
    except (TypeError, ValueError): return (1, str(r["value"]))
rows.sort(key=_key)

xs = [r["value"] for r in rows]
def col(k): return [r[k] for r in rows]
m50, m5095, kern = col("map50"), col("map50_95"), col("kernel_ms")


def _xnum(xs):
    try: return [float(x) for x in xs], True
    except (TypeError, ValueError): return list(range(len(xs))), False


xn, numeric = _xnum(xs)

# ---- accuracy-vs-value plot ----
plt.figure(figsize=(6, 4))
if any(v is not None for v in m5095):
    plt.plot(xn, m5095, "o-", label="mAP50-95")
if any(v is not None for v in m50):
    plt.plot(xn, m50, "s--", alpha=0.55, label="mAP50")
if not numeric:
    plt.xticks(xn, [str(x) for x in xs], rotation=30, ha="right")
plt.xlabel(KNOB); plt.ylabel("mAP"); plt.title(f"Accuracy vs {KNOB}")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(OUTDIR / "accuracy_vs_value.png", dpi=120); plt.close()

# ---- latency-vs-value plot ----
plt.figure(figsize=(6, 4))
kv = [k for k in kern if isinstance(k, (int, float))]
plt.plot(xn, kern, "o-", color="C3")
if not numeric:
    plt.xticks(xn, [str(x) for x in xs], rotation=30, ha="right")
plt.xlabel(KNOB); plt.ylabel("kernel ms"); plt.title(f"Latency vs {KNOB}")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUTDIR / "latency_vs_value.png", dpi=120); plt.close()

flat = len(kv) >= 2 and (max(kv) - min(kv) < 0.05)
lat_note = (f"**Latency invariant to this knob** — same engine structure "
            f"(range {min(kv):.3f}–{max(kv):.3f} ms, <0.05 ms spread)."
            if flat else "Latency varies across the sweep — see plot.") if kv else \
           "Latency not recorded."


def f(x, d=4): return f"{x:.{d}f}" if isinstance(x, (int, float)) else "—"


# best-accuracy row
best = max((r for r in rows if isinstance(r["map50_95"], (int, float))),
           key=lambda r: r["map50_95"], default=None)
best_line = (f"**Peak mAP50-95 = {best['map50_95']:.4f} at {KNOB}={best['value']}** "
             f"({best['version']})." if best else "")

table = "\n".join(
    f"| {r['value']} | {f(r['map50'])} | {f(r['map50_95'])} | {f(r['kernel_ms'],3)} | {r['version']} |"
    for r in rows)

md = f"""# {KNOB} sweep

Non-binary knob sweep — value → accuracy → latency. {len(rows)} runs. All other knobs at baseline.

{best_line}

## Results
| {KNOB} | mAP50 | mAP50-95 | kernel ms | version |
|---|---|---|---|---|
{table}

## Accuracy vs value
![accuracy](accuracy_vs_value.png)

Shows the SHAPE — where mAP peaks, whether it's monotonic, plateaus, or degrades.

## Latency vs value
![latency](latency_vs_value.png)

{lat_note}
"""
(OUTDIR / "results.md").write_text(md)
print(f"[sweep-doc] wrote {OUTDIR/'results.md'} + accuracy_vs_value.png + latency_vs_value.png")
print(f"[sweep-doc] {len(rows)} points; latency {'FLAT (invariant)' if flat else 'varies'}")

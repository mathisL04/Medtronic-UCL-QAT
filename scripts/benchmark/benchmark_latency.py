from pathlib import Path
import csv
import gc
import os
import socket
import sys
import time

import numpy as np
import cv2
import torch
import pynvml
from ultralytics import YOLO


# ==========================================================
# BENCHMARK CONFIGURATION
#
# MODIFY when changing model / dataset / benchmark protocol:
# - DEVICE
# - IMG_SIZE
# - CONF
# - BENCHMARK_REPEATS
# - WARMUP_IMAGES
# - MODEL_PATH
# - IMG_DIR
# - OUT_DIR
#
# CONF=0.25 is appropriate for deployment-style latency.
# Do NOT confuse this with conf=0.001 used for mAP evaluation.
# ==========================================================

if "DEVICE" not in os.environ:
    sys.exit(
        "DEVICE is not set. Refusing to guess a GPU.\n"
        "Set explicitly, e.g. DEVICE=0 python scripts/benchmark/benchmark_latency.py"
    )

DEVICE = int(os.environ["DEVICE"])

SEED = int(os.environ.get("SEED", 42))

IMG_SIZE = int(
    os.environ.get("IMG_SIZE", 640)
)

CONF = float(
    os.environ.get("CONF", 0.25)
)

BENCHMARK_REPEATS = int(
    os.environ.get(
        "BENCHMARK_REPEATS",
        5,
    )
)

WARMUP_IMAGES = int(
    os.environ.get(
        "WARMUP_IMAGES",
        30,
    )
)


# MODIFY only if changing the GPU-idle acceptance criteria.
GATE_UTIL_THRESHOLD = int(
    os.environ.get(
        "GATE_UTIL_THRESHOLD",
        10,
    )
)

GATE_SAMPLES = int(
    os.environ.get(
        "GATE_SAMPLES",
        5,
    )
)

GATE_INTERVAL_S = float(
    os.environ.get(
        "GATE_INTERVAL_S",
        0.2,
    )
)


# MODIFY: checkpoint of the model to benchmark.
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
        "models/yolo26n_sanoscience_full_left/0_baseline_pytorch/best.pt",
    )
)


# MODIFY: benchmark/evaluation images.
#
# Keep the same image set when comparing several models/precisions.
IMG_DIR = Path(
    os.environ.get(
        "IMG_DIR",
        "/home/zcemml1/medtronic_qat_data/"
        "demo_val100_random_yolo/images/val",
    )
)


# MODIFY: where CSV benchmark results are written.
OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        "/home/zcemml1/medtronic_qat_data/runs_sanoscience",
    )
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HOST = socket.gethostname()
OWN_PID = os.getpid()


def clean_runtime():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def resolve_physical_index(
    torch_device,
):

    cvd = os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )

    if (
        cvd is None
        or cvd.strip() == ""
    ):
        return torch_device

    visible = [
        x.strip()
        for x in cvd.split(",")
        if x.strip() != ""
    ]

    if torch_device >= len(visible):
        sys.exit(
            f"DEVICE={torch_device} "
            f"is out of range for "
            f"CUDA_VISIBLE_DEVICES={cvd!r}."
        )

    return int(
        visible[torch_device]
    )


def _compute_pids(
    handle,
):

    procs = (
        pynvml
        .nvmlDeviceGetComputeRunningProcesses(
            handle
        )
    )

    return [
        (
            p.pid,
            p.usedGpuMemory,
        )
        for p in procs
    ]


def _print_procs(
    procs,
):

    for pid, mem in procs:

        mib = (
            "unknown"
            if mem is None
            else
            f"{mem / (1024 ** 2):.0f} MiB"
        )

        print(
            f"  pid {pid}: {mib}"
        )


# ==========================================================
# GPU IDLE GATE
#
# Usually leave unchanged.
#
# MODIFY only if the new hardware / cluster environment
# requires a different contention policy.
# ==========================================================

def gate_gpu_or_exit(
    physical_index,
):

    pynvml.nvmlInit()

    try:

        count = (
            pynvml
            .nvmlDeviceGetCount()
        )

        if physical_index >= count:
            sys.exit(
                f"Physical GPU index "
                f"{physical_index} "
                f"does not exist."
            )

        handle = (
            pynvml
            .nvmlDeviceGetHandleByIndex(
                physical_index
            )
        )

        name = (
            pynvml
            .nvmlDeviceGetName(
                handle
            )
        )

        if isinstance(
            name,
            bytes,
        ):
            name = name.decode()

        print(
            "------------------------------------------------------------"
        )
        print("GPU gate")
        print(
            "------------------------------------------------------------"
        )
        print(
            f"Target physical GPU "
            f"{physical_index}: {name}"
        )

        busy = _compute_pids(
            handle
        )

        if busy:

            print(
                "Refusing to run. "
                "Compute processes already "
                "on this GPU:"
            )

            _print_procs(
                busy
            )

            sys.exit(1)


        util_samples = []

        for i in range(
            GATE_SAMPLES
        ):

            util = (
                pynvml
                .nvmlDeviceGetUtilizationRates(
                    handle
                )
            )

            util_samples.append(
                util.gpu
            )

            busy = _compute_pids(
                handle
            )

            if busy:

                print(
                    "Refusing to run. "
                    "A compute process appeared "
                    "during the gate:"
                )

                _print_procs(
                    busy
                )

                sys.exit(1)


            if (
                i
                < GATE_SAMPLES - 1
            ):
                time.sleep(
                    GATE_INTERVAL_S
                )


        mem = (
            pynvml
            .nvmlDeviceGetMemoryInfo(
                handle
            )
        )

        peak_util = max(
            util_samples
        )


        print(
            "Compute processes: none"
        )

        print(
            f"Utilisation samples (%): "
            f"{util_samples} "
            f"peak={peak_util}"
        )

        print(
            f"Memory used: "
            f"{mem.used / (1024 ** 2):.0f} MiB "
            f"of "
            f"{mem.total / (1024 ** 2):.0f} MiB"
        )


        if (
            peak_util
            > GATE_UTIL_THRESHOLD
        ):
            sys.exit(
                f"Refusing to run. "
                f"Peak utilisation "
                f"{peak_util}% exceeds "
                f"GATE_UTIL_THRESHOLD="
                f"{GATE_UTIL_THRESHOLD}%."
            )


        print(
            "Gate passed: GPU is idle."
        )
        print()


    finally:

        pynvml.nvmlShutdown()


def gpu_snapshot(
    physical_index,
    own_pid,
):

    pynvml.nvmlInit()

    try:

        handle = (
            pynvml
            .nvmlDeviceGetHandleByIndex(
                physical_index
            )
        )

        util = (
            pynvml
            .nvmlDeviceGetUtilizationRates(
                handle
            )
        )

        mem = (
            pynvml
            .nvmlDeviceGetMemoryInfo(
                handle
            )
        )

        procs = (
            pynvml
            .nvmlDeviceGetComputeRunningProcesses(
                handle
            )
        )

        others = [
            p.pid
            for p in procs
            if p.pid != own_pid
        ]

        return {
            "gpu_util_pct":
                int(util.gpu),

            "mem_used_mib":
                int(
                    round(
                        mem.used
                        / (1024 ** 2)
                    )
                ),

            "compute_procs_other":
                len(others),
        }

    finally:

        pynvml.nvmlShutdown()


def summarize(
    values,
):

    arr = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean_ms":
            float(arr.mean()),

        "std_ms":
            (
                float(
                    arr.std(
                        ddof=1
                    )
                )
                if arr.size > 1
                else 0.0
            ),

        "median_ms":
            float(
                np.median(arr)
            ),

        "min_ms":
            float(arr.min()),

        "p95_ms":
            float(
                np.percentile(
                    arr,
                    95,
                )
            ),

        "p99_ms":
            float(
                np.percentile(
                    arr,
                    99,
                )
            ),

        "max_ms":
            float(arr.max()),
    }


PHYSICAL_INDEX = (
    resolve_physical_index(
        DEVICE
    )
)

gate_gpu_or_exit(
    PHYSICAL_INDEX
)


# ==========================================================
# RUNTIME
#
# Usually leave unchanged.
#
# MODIFY if benchmarking a non-CUDA device or if the
# new model requires special backend settings.
# ==========================================================

torch.set_num_threads(1)

if torch.cuda.is_available():
    torch.cuda.set_device(
        DEVICE
    )

torch.backends.cudnn.benchmark = True

clean_runtime()


print(
    "============================================================"
)
print(
    "PyTorch latency benchmark"
)
print(
    "============================================================"
)

print("Host:", HOST)

print(
    "CUDA_VISIBLE_DEVICES:",
    os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ),
)

print(
    "PyTorch visible device:",
    DEVICE,
)

print(
    "Physical NVML index:",
    PHYSICAL_INDEX,
)

if torch.cuda.is_available():
    print(
        "CUDA device name:",
        torch.cuda.get_device_name(
            DEVICE
        ),
    )

print(
    "Model:",
    MODEL_PATH,
)

print(
    "Image directory:",
    IMG_DIR,
)

print(
    "Image size:",
    IMG_SIZE,
)

print(
    "Confidence:",
    CONF,
)

print(
    "Repeats:",
    BENCHMARK_REPEATS,
)

print(
    "Warmup images:",
    WARMUP_IMAGES,
)


image_paths = sorted(
    IMG_DIR.glob("*.jpg")
)

if not image_paths:
    raise ValueError(
        f"No images found in "
        f"{IMG_DIR}"
    )


print(
    "Benchmark images:",
    len(image_paths),
)


# ==========================================================
# IMAGE LOADING
#
# MODIFY if the new dataset uses another file type
# or requires another decoder.
#
# Disk access is intentionally excluded from timing.
# ==========================================================

print(
    "\nPreloading images into RAM..."
)

ram_images = []

for path in image_paths:

    img = cv2.imread(
        str(path)
    )

    if img is None:
        raise ValueError(
            f"Could not read image: "
            f"{path}"
        )

    ram_images.append(
        (
            path.name,
            img,
        )
    )


print(
    f"Loaded "
    f"{len(ram_images)} "
    f"images into RAM."
)


# ==========================================================
# MODEL LOADING
#
# MODIFY this section if adapting the benchmark to a model
# that is not loadable through Ultralytics YOLO().
# ==========================================================

print(
    "\nLoading model once..."
)

model = YOLO(
    str(MODEL_PATH)
)

model.model.eval()

clean_runtime()

print(
    "Model loaded."
)


# ==========================================================
# WARMUP
#
# Keep identical between model/precision comparisons.
# ==========================================================

print(
    "\nWarmup..."
)

warmup_list = ram_images[
    :min(
        WARMUP_IMAGES,
        len(ram_images),
    )
]

with torch.inference_mode():

    for _, img in warmup_list:

        _ = model.predict(
            source=img,
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,
            save=False,
            verbose=False,
        )


if torch.cuda.is_available():
    torch.cuda.synchronize()


print(
    "Warmup complete."
)


# ==========================================================
# BENCHMARK
#
# Current timing decomposition uses Ultralytics:
#
# preprocess
# inference
# postprocess
# total
#
# MODIFY this block if another model/framework does not
# expose result.speed in the same format.
# ==========================================================

pool = {
    "preprocess_ms": [],
    "inference_ms": [],
    "postprocess_ms": [],
    "total_ms": [],
}


per_repeat = []


snapshot_at = {
    len(ram_images) // 4,
    len(ram_images) // 2,
    (
        3 * len(ram_images)
    ) // 4,
}


for run_idx in range(
    1,
    BENCHMARK_REPEATS + 1,
):

    print(
        f"\nRun "
        f"{run_idx}/"
        f"{BENCHMARK_REPEATS}"
    )

    clean_runtime()

    rows = []
    snaps = []


    with torch.inference_mode():

        for i, (
            image_name,
            img,
        ) in enumerate(
            ram_images
        ):

            result = model.predict(
                source=img,
                imgsz=IMG_SIZE,
                conf=CONF,
                device=DEVICE,
                save=False,
                verbose=False,
            )[0]


            pre = (
                result.speed[
                    "preprocess"
                ]
            )

            inf = (
                result.speed[
                    "inference"
                ]
            )

            post = (
                result.speed[
                    "postprocess"
                ]
            )

            total = (
                pre
                + inf
                + post
            )


            boxes = (
                len(result.boxes)
                if result.boxes
                is not None
                else 0
            )


            rows.append({
                "run":
                    run_idx,

                "image":
                    image_name,

                "preprocess_ms":
                    pre,

                "inference_ms":
                    inf,

                "postprocess_ms":
                    post,

                "total_ms":
                    total,

                "num_boxes":
                    boxes,
            })


            pool[
                "preprocess_ms"
            ].append(pre)

            pool[
                "inference_ms"
            ].append(inf)

            pool[
                "postprocess_ms"
            ].append(post)

            pool[
                "total_ms"
            ].append(total)


            if i in snapshot_at:

                snaps.append(
                    gpu_snapshot(
                        PHYSICAL_INDEX,
                        OWN_PID,
                    )
                )


    if torch.cuda.is_available():
        torch.cuda.synchronize()


    run_total = np.asarray(
        [
            r["total_ms"]
            for r in rows
        ],
        dtype=float,
    )


    util_peak = max(
        (
            s["gpu_util_pct"]
            for s in snaps
        ),
        default=-1,
    )

    mem_peak = max(
        (
            s["mem_used_mib"]
            for s in snaps
        ),
        default=-1,
    )

    others_peak = max(
        (
            s[
                "compute_procs_other"
            ]
            for s in snaps
        ),
        default=0,
    )


    per_repeat.append({
        "run":
            run_idx,

        "mean_total_ms":
            float(
                run_total.mean()
            ),

        "min_total_ms":
            float(
                run_total.min()
            ),

        "max_total_ms":
            float(
                run_total.max()
            ),

        "gpu_util_peak_pct":
            util_peak,

        "mem_used_peak_mib":
            mem_peak,

        "compute_procs_other_peak":
            others_peak,
    })


    flag = (
        ""
        if others_peak == 0
        else
        f" <-- CONTENDED "
        f"({others_peak} other proc)"
    )


    print(
        f"  images: {len(rows)} "
        f"  mean total: "
        f"{run_total.mean():.3f} ms "
        f"  min: "
        f"{run_total.min():.3f} "
        f"  max: "
        f"{run_total.max():.3f}"
    )


    print(
        f"  gpu snapshot: "
        f"util_peak={util_peak}% "
        f"mem_peak={mem_peak} MiB "
        f"other_compute_procs="
        f"{others_peak}{flag}"
    )


    out_csv = (
        OUT_DIR
        / (
            f"benchmark_latency_"
            f"seed{SEED}_"
            f"run{run_idx}.csv"
        )
    )


    with open(
        out_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ==========================================================
# SUMMARY
#
# Usually leave unchanged.
#
# Reports pooled:
# mean, std, median, min, p95, p99, max.
# ==========================================================

n_pooled = len(
    pool["total_ms"]
)


stage_order = [
    (
        "preprocess_ms",
        "Preprocess",
    ),
    (
        "inference_ms",
        "Inference",
    ),
    (
        "postprocess_ms",
        "Postprocess",
    ),
    (
        "total_ms",
        "Total",
    ),
]


summary_rows = []


for key, label in stage_order:

    s = {
        "stage": label,
        "n": n_pooled,
    }

    s.update(
        summarize(
            pool[key]
        )
    )

    summary_rows.append(
        s
    )


fps = (
    1000.0
    / summary_rows[-1][
        "mean_ms"
    ]
)


summary_csv = (
    OUT_DIR
    / (
        f"benchmark_latency_"
        f"seed{SEED}_"
        f"pooled_summary.csv"
    )
)


with open(
    summary_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            summary_rows[0].keys(),
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


print(
    "\n============================================================"
)
print(
    "Pooled latency summary"
)
print(
    "============================================================"
)

print(
    f"Runs: "
    f"{BENCHMARK_REPEATS} "
    f"Images/run: "
    f"{len(ram_images)} "
    f"Pooled samples: "
    f"{n_pooled}"
)

print(
    "CSV summary:",
    summary_csv,
)


header = (
    f"{'Stage':<12} "
    f"{'Mean':>9} "
    f"{'Std':>9} "
    f"{'Median':>9} "
    f"{'Min':>9} "
    f"{'P95':>9} "
    f"{'P99':>9} "
    f"{'Max':>9}"
)

print(header)
print(
    "-" * len(header)
)


for row in summary_rows:

    print(
        f"{row['stage']:<12} "
        f"{row['mean_ms']:>9.3f} "
        f"{row['std_ms']:>9.3f} "
        f"{row['median_ms']:>9.3f} "
        f"{row['min_ms']:>9.3f} "
        f"{row['p95_ms']:>9.3f} "
        f"{row['p99_ms']:>9.3f} "
        f"{row['max_ms']:>9.3f}"
    )


print(
    "-" * len(header)
)

print(
    f"Approx FPS "
    f"(from pooled Total mean): "
    f"{fps:.2f}"
)


repeats_csv = (
    OUT_DIR
    / (
        f"benchmark_latency_"
        f"seed{SEED}_"
        f"per_repeat.csv"
    )
)


with open(
    repeats_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            per_repeat[0].keys(),
    )

    writer.writeheader()

    writer.writerows(
        per_repeat
    )


contended = [
    r["run"]
    for r in per_repeat
    if (
        r[
            "compute_procs_other_peak"
        ]
        > 0
    )
]


if contended:

    print(
        f"WARNING: contention "
        f"detected on repeat(s): "
        f"{contended}"
    )

else:

    print(
        "No other compute "
        "processes seen on "
        "any repeat."
    )


print(
    "Done."
)

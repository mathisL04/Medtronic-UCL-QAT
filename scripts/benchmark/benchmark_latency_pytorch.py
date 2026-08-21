import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


# ==========================================================
# BENCHMARK CONFIGURATION
#
# MODIFY when changing model / dataset / benchmark protocol:
# - MODEL_PATH
# - QAT_STATE
# - IMG_DIR
# - OUT_DIR
# - MODEL_MODE
# - IMG_SIZE
# - BENCHMARK_REPEATS
# - WARMUP_IMAGES
# - DEVICE
#
# MODEL_MODE:
#   fp32 -> baseline PyTorch model
#   fp16 -> baseline converted to PyTorch FP16
#   qat  -> restored ModelOpt fake-quant model
# ==========================================================

REPO = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT"
)


# MODIFY: baseline PyTorch checkpoint.
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(
            REPO
            / "models/yolo26n_sanoscience_full_left/"
              "baseline/best.pt"
        ),
    )
)


# MODIFY: trained ModelOpt QAT state.
# Required only when MODEL_MODE=qat.
QAT_STATE = Path(
    os.environ.get(
        "QAT_STATE",
        str(
            REPO
            / "models/yolo26n_sanoscience_full_left/"
              "qat/v6_final/qat_modelopt_state_best.pt"
        ),
    )
)


# MODIFY: benchmark image set.
#
# Keep this identical when comparing FP32 / FP16 / QAT.
IMG_DIR = Path(
    os.environ.get(
        "IMG_DIR",
        "/home/zcemml1/medtronic_qat_data/"
        "demo_val100_random_yolo/images/val",
    )
)


# MODIFY: output directory for benchmark provenance.
OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        "/home/zcemml1/medtronic_qat_data/"
        "runs_sanoscience",
    )
)


MODEL_MODE = os.environ.get(
    "MODEL_MODE",
    "",
).lower()

if MODEL_MODE not in (
    "fp32",
    "fp16",
    "qat",
):
    sys.exit(
        "MODEL_MODE must be "
        "fp32 | fp16 | qat"
    )


SEED = int(
    os.environ.get(
        "SEED",
        42,
    )
)


# MODIFY if the replacement model uses another input size.
IMG_SIZE = int(
    os.environ.get(
        "IMG_SIZE",
        640,
    )
)


BENCHMARK_REPEATS = int(
    os.environ.get(
        "BENCHMARK_REPEATS",
        10,
    )
)


WARMUP_IMAGES = int(
    os.environ.get(
        "WARMUP_IMAGES",
        50,
    )
)


# MODIFY only for a different shared-GPU policy.
GATE_ALLOW_IDLE_MIB = int(
    os.environ.get(
        "GATE_ALLOW_IDLE_MIB",
        0,
    )
)


# MODIFY: GPU to benchmark.
DEVICE = os.environ.get(
    "DEVICE"
)

if DEVICE is None:
    sys.exit(
        "DEVICE is required "
        "(e.g. DEVICE=2). "
        "No GPU specified, no run."
    )


# ==========================================================
# GPU IDLE GATE
#
# Usually leave unchanged.
# Modify only for a different cluster/GPU-sharing policy.
# ==========================================================

import pynvml  # noqa: E402

pynvml.nvmlInit()

_procs = (
    pynvml
    .nvmlDeviceGetComputeRunningProcesses(
        pynvml
        .nvmlDeviceGetHandleByIndex(
            int(DEVICE)
        )
    )
)

_foreign = [
    (
        p.pid,
        (p.usedGpuMemory or 0)
        // (1 << 20),
    )
    for p in _procs
]

_blocking = [
    pp
    for pp in _foreign
    if pp[1] > GATE_ALLOW_IDLE_MIB
]

exclusive = (
    len(_foreign) == 0
)

if _blocking:
    sys.exit(
        f"Gate FAILED: other compute "
        f"processes on GPU {DEVICE}: "
        f"{_blocking}"
    )

print(
    f"Gate passed "
    f"(device {DEVICE}, "
    f"foreign {_foreign}, "
    f"exclusive={exclusive})"
)


os.environ[
    "CUDA_VISIBLE_DEVICES"
] = DEVICE


import cv2  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def summarize(v):

    a = np.asarray(
        v,
        dtype=float,
    )

    return {
        "n":
            int(a.size),

        "mean_ms":
            float(a.mean()),

        "std_ms":
            (
                float(
                    a.std(
                        ddof=1
                    )
                )
                if a.size > 1
                else 0.0
            ),

        "median_ms":
            float(
                np.median(a)
            ),

        "min_ms":
            float(a.min()),

        "p95_ms":
            float(
                np.percentile(
                    a,
                    95,
                )
            ),

        "p99_ms":
            float(
                np.percentile(
                    a,
                    99,
                )
            ),

        "max_ms":
            float(a.max()),
    }


# ==========================================================
# PREPROCESSING
#
# MODIFY if the replacement model uses different:
# - resize / crop behaviour
# - normalization
# - channel order
# - input tensor layout
#
# This preprocessing must match the model's actual inference path.
# ==========================================================

def letterbox(
    img,
    s,
):

    h0, w0 = img.shape[:2]

    r = min(
        s / h0,
        s / w0,
    )

    nw = int(
        round(
            w0 * r
        )
    )

    nh = int(
        round(
            h0 * r
        )
    )

    canvas = np.full(
        (
            s,
            s,
            3,
        ),
        114,
        dtype=np.uint8,
    )

    top = (
        s - nh
    ) // 2

    left = (
        s - nw
    ) // 2

    canvas[
        top:top + nh,
        left:left + nw,
    ] = cv2.resize(
        img,
        (
            nw,
            nh,
        ),
        interpolation=
            cv2.INTER_LINEAR,
    )

    return canvas


def preprocess(
    img,
    s,
):

    im = (
        letterbox(
            img,
            s,
        )[:, :, ::-1]
        .transpose(
            2,
            0,
            1,
        )
    )

    return (
        np.ascontiguousarray(
            im,
            dtype=np.float32,
        )[None]
        / 255.0
    )


print(
    "=" * 60
)

print(
    f"PyTorch latency study "
    f"-- MODE={MODEL_MODE} "
    f"(raw eager, no conversion)"
)

print(
    "=" * 60
)


# ==========================================================
# MODEL LOADING
#
# MODIFY this function if the replacement architecture
# cannot be loaded with Ultralytics YOLO().
#
# For MODEL_MODE=qat, the QAT state must be restored onto
# the corresponding baseline architecture.
# ==========================================================

def load_base():

    y = YOLO(
        str(MODEL_PATH)
    )

    if MODEL_MODE == "qat":

        import modelopt.torch.opt as mto

        mto.restore(
            y.model,
            str(QAT_STATE),
        )

    return y


yA = load_base()

model = (
    yA.model
    .eval()
    .cuda()
)

for p in model.parameters():
    p.requires_grad = False


# ==========================================================
# PYTORCH FP16 MODE
#
# Usually leave unchanged.
#
# If the replacement model cannot run under model.half(),
# the script falls back to autocast and records that the
# FP16 benchmark contains FP32 islands.
# ==========================================================

half_input = False
fp16_mode = None
use_autocast = False


if MODEL_MODE == "fp16":

    test = torch.from_numpy(
        preprocess(
            np.zeros(
                (
                    IMG_SIZE,
                    IMG_SIZE,
                    3,
                ),
                np.uint8,
            ),
            IMG_SIZE,
        )
    )

    try:

        model = model.half()

        with torch.no_grad():

            model(
                test
                .half()
                .cuda()
            )

        fp16_mode = (
            "pure model.half()"
        )

        half_input = True

        print(
            "[fp16] pure model.half() "
            "forward OK -> clean FP16"
        )

    except Exception as e:

        model = (
            yA.model
            .eval()
            .float()
            .cuda()
        )

        fp16_mode = (
            "autocast (FP32 islands)"
        )

        use_autocast = True

        print(
            f"[fp16] model.half() "
            f"forward FAILED "
            f"({type(e).__name__}: "
            f"{str(e)[:120]})"
        )

        print(
            "[fp16] falling back "
            "to torch.autocast(fp16)"
        )


print(
    "torch",
    torch.__version__,
    "| cuda",
    torch.cuda.get_device_name(
        0
    ),
)


def forward(x):

    with torch.no_grad():

        if use_autocast:

            with torch.autocast(
                "cuda",
                dtype=torch.float16,
            ):
                return model(x)

        return model(x)


# ==========================================================
# DATASET LOADING
#
# MODIFY the glob/decoder if the replacement dataset uses
# another image format or data representation.
# ==========================================================

paths = sorted(
    IMG_DIR.glob(
        "*.jpg"
    )
)

if not paths:
    sys.exit(
        f"No images in "
        f"{IMG_DIR}"
    )


ram = [
    (
        p.name,
        cv2.imread(
            str(p)
        ),
    )
    for p in paths
]


print(
    f"preloaded "
    f"{len(ram)} images"
)


# ==========================================================
# COLUMN A:
# PURE GPU FORWARD LATENCY
#
# CUDA events measure only the raw PyTorch forward pass.
#
# Keep this method identical across FP32 / FP16 / QAT.
# ==========================================================

for _, img in ram[
    :min(
        WARMUP_IMAGES,
        len(ram),
    )
]:

    x = (
        torch
        .from_numpy(
            preprocess(
                img,
                IMG_SIZE,
            )
        )
        .cuda()
    )

    forward(
        x.half()
        if half_input
        else x
    )


torch.cuda.synchronize()


ev_s = torch.cuda.Event(
    enable_timing=True
)

ev_e = torch.cuda.Event(
    enable_timing=True
)


fwd_ms = []


for run in range(
    BENCHMARK_REPEATS
):

    for _, img in ram:

        x = (
            torch
            .from_numpy(
                preprocess(
                    img,
                    IMG_SIZE,
                )
            )
            .cuda()
        )

        if half_input:
            x = x.half()

        torch.cuda.synchronize()

        ev_s.record()

        forward(x)

        ev_e.record()

        torch.cuda.synchronize()

        fwd_ms.append(
            ev_s.elapsed_time(
                ev_e
            )
        )


A = summarize(
    fwd_ms
)


print(
    f"\n[A] Forward(GPU) CUDA-event: "
    f"median "
    f"{A['median_ms']:.3f} ms "
    f"(mean "
    f"{A['mean_ms']:.3f}) "
    f"-> "
    f"{1000.0 / A['median_ms']:.1f} FPS"
)


# ==========================================================
# COLUMN B:
# ULTRALYTICS END-TO-END PIPELINE
#
# MODIFY this block if the replacement model is not an
# Ultralytics YOLO model or does not expose result.speed.
# ==========================================================

yB = load_base()

half_flag = (
    MODEL_MODE == "fp16"
)


yB.predict(
    ram[0][1],
    device=0,
    half=half_flag,
    verbose=False,
    imgsz=IMG_SIZE,
)


pre = []
inf = []
post = []


for run in range(
    BENCHMARK_REPEATS
):

    for _, img in ram:

        r = yB.predict(
            img,
            device=0,
            half=half_flag,
            verbose=False,
            imgsz=IMG_SIZE,
        )

        sp = r[0].speed

        pre.append(
            sp["preprocess"]
        )

        inf.append(
            sp["inference"]
        )

        post.append(
            sp["postprocess"]
        )


tot = [
    a + b + c
    for a, b, c
    in zip(
        pre,
        inf,
        post,
    )
]


Bpre = summarize(pre)
Binf = summarize(inf)
Bpost = summarize(post)
Btot = summarize(tot)


print(
    f"[B] Ultralytics pipeline: "
    f"pre "
    f"{Bpre['median_ms']:.3f} "
    f"+ inf "
    f"{Binf['median_ms']:.3f} "
    f"+ post "
    f"{Bpost['median_ms']:.3f} "
    f"= total "
    f"{Btot['median_ms']:.3f} ms "
    f"(half={half_flag}) "
    f"-> "
    f"{1000.0 / Btot['median_ms']:.1f} FPS"
)


# ==========================================================
# OUTPUT / PROVENANCE
#
# Usually leave unchanged.
# ==========================================================

prov = {
    "timestamp_utc":
        datetime
        .now(
            timezone.utc
        )
        .isoformat(),

    "stage":
        "pytorch_raw_latency",

    "model_mode":
        MODEL_MODE,

    "note":
        (
            "eager PyTorch, no conversion. "
            "qat=fake-quant simulation "
            "(not real INT8). "
            "fp16 cleanliness: "
            + str(fp16_mode)
        ),

    "fp16_mode":
        fp16_mode,

    "device":
        DEVICE,

    "exclusive_gpu":
        exclusive,

    "model_path":
        str(MODEL_PATH),

    "qat_state":
        (
            str(QAT_STATE)
            if MODEL_MODE == "qat"
            else None
        ),

    "img_dir":
        str(IMG_DIR),

    "n_images":
        len(ram),

    "repeats":
        BENCHMARK_REPEATS,

    "img_size":
        IMG_SIZE,

    "seed":
        SEED,

    "torch":
        torch.__version__,

    "A_forward_gpu":
        A,

    "B_pipeline": {
        "preprocess":
            Bpre,

        "inference":
            Binf,

        "postprocess":
            Bpost,

        "total":
            Btot,

        "half":
            half_flag,
    },

    "fps_A_forward":
        1000.0
        / A["median_ms"],

    "fps_B_total":
        1000.0
        / Btot["median_ms"],
}


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


out = (
    OUT_DIR
    / (
        f"benchmark_latency_pytorch_"
        f"{MODEL_MODE}_"
        f"seed{SEED}_"
        f"summary.provenance.json"
    )
)


out.write_text(
    json.dumps(
        prov,
        indent=2,
    )
)


print(
    "Provenance:",
    out,
)

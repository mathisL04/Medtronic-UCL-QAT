#!/usr/bin/env bash
set -o pipefail


# ==========================================================
# USER CONFIGURATION
# Modify this section when changing repository location,
# Python environment, storage layout, training script,
# hardware selection or run parameters.
# ==========================================================

# Project repository.
REPO=/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT

# Python environment used for QAT.
# Override with PY=/path/to/python when changing environments.
PY="${PY:-/home/zcemml1/venvs/medtronic-qat/bin/python}"

# Experiment controls.
# RUN_NAME and DEVICE are intentionally required.
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
DEVICE="${DEVICE:?DEVICE is required}"
EPOCHS="${EPOCHS:-1}"

# Optional QAT training parameters passed through to train_qat.py.
PATIENCE="${PATIENCE:-0}"
BATCH="${BATCH:-16}"
IMG_SIZE="${IMG_SIZE:-640}"
LR0="${LR0:-0.001}"
LRF="${LRF:-0.01}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEED="${CALIB_SEED:-42}"
WORKERS="${WORKERS:-4}"

# Local temporary training location.
# Change when another machine uses a different fast/local scratch filesystem.
SCRATCH="${SCRATCH:-/tmp/zcemml1_qat}"

# Persistent output location.
# Change when using another cluster, filesystem or experiment structure.
NFS_OUT="${NFS_OUT:-$REPO/runs_qat/$RUN_NAME}"

# QAT training script.
# Override when testing another trainer implementation.
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/train_qat.py}"

# Minimum free-space requirements in MB.
# Adapt to the filesystem and expected experiment size.
MIN_SCRATCH_MB="${MIN_SCRATCH_MB:-5000}"
MIN_DURABLE_MB="${MIN_DURABLE_MB:-1000}"


stamp() {
    date '+%Y-%m-%d %H:%M:%S'
}


cd "$REPO" || exit 1

mkdir -p \
    "$SCRATCH" \
    "$NFS_OUT"


echo "[$(stamp)] QAT run: $RUN_NAME"
echo "[$(stamp)] device: $DEVICE"
echo "[$(stamp)] epochs: $EPOCHS"
echo "[$(stamp)] scratch: $SCRATCH"
echo "[$(stamp)] output: $NFS_OUT"


# Environment-specific storage check.
# Modify/remove the quota command if the target system does not use NFS quota.
quota_free_mb=$(
    quota -s 2>/dev/null \
    | grep -A1 "vol_home_2/zcemml1" \
    | tail -1 \
    | awk '{print int($3)-int($1)}'
)

tmp_free_mb=$(
    df -m "$SCRATCH" \
    | tail -1 \
    | awk '{print $4}'
)


echo "[$(stamp)] durable free: ${quota_free_mb:-?} MB"
echo "[$(stamp)] scratch free: ${tmp_free_mb:-?} MB"


if [ "${tmp_free_mb:-0}" -lt "$MIN_SCRATCH_MB" ]; then
    echo "[$(stamp)] ABORT: insufficient scratch space."
    exit 1
fi


if [ "${quota_free_mb:-0}" -lt "$MIN_DURABLE_MB" ]; then
    echo "[$(stamp)] ABORT: insufficient durable storage."
    exit 1
fi


copy_back() {
    rc=$?

    src="$SCRATCH/$RUN_NAME"

    echo "[$(stamp)] copy-back (exit code $rc)"

    if [ -d "$src" ]; then

        # Modify this list if train_qat.py produces additional
        # artifacts that should be kept permanently.
        for f in \
            qat_modelopt_state.pt \
            qat_modelopt_state_best.pt \
            qat_provenance.json \
            results.csv \
            args.yaml
        do
            if [ -f "$src/$f" ]; then
                cp -p \
                    "$src/$f" \
                    "$NFS_OUT/"

                echo \
                    "[$(stamp)] saved $f"
            fi
        done

    else
        echo \
            "[$(stamp)] no run directory found at $src"
    fi


    echo "[$(stamp)] durable artifacts:"

    ls -la "$NFS_OUT" 2>/dev/null \
        | tail -n +2 \
        | awk \
            '{print "             "$5"  "$9}'


    exit "$rc"
}


trap copy_back EXIT


DEVICE="$DEVICE" \
EPOCHS="$EPOCHS" \
PATIENCE="$PATIENCE" \
BATCH="$BATCH" \
IMG_SIZE="$IMG_SIZE" \
LR0="$LR0" \
LRF="$LRF" \
N_CALIB="$N_CALIB" \
CALIB_SEED="$CALIB_SEED" \
WORKERS="$WORKERS" \
RUN_NAME="$RUN_NAME" \
PROJECT="$SCRATCH" \
"$PY" -u "$TRAIN_SCRIPT"

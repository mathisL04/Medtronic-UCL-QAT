#!/usr/bin/env bash
# Run a QAT job with checkpoints on LOCAL scratch, then copy the durable
# artifacts back to NFS before exiting.
#
# WHY: the NFS home has a 50 GB quota. Ultralytics writes last.pt + best.pt every
# epoch plus plots; that churn on NFS is what exhausted the quota once already,
# and a full-quota checkpoint write fails as an mmap segfault three steps later
# rather than as a disk error -- which cost an afternoon to diagnose.
#
# /tmp is /dev/sda2, local, outside the quota, and ~3.5x faster than NFS. It is
# also NODE-LOCAL and NOT durable: cleared on reboot and invisible from Cork. So
# the copy-back is part of this script, not a manual afterthought -- a finished
# QAT model must never live only on Geneva's /tmp.
#
# Usage:  DEVICE=1 EPOCHS=1 RUN_NAME=qat_smoke bash scripts/qat_run.sh
set -o pipefail

REPO=/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT
# Interpreter is overridable so the QAT stack can migrate venvs (e.g. the
# Py3.11 + modelopt-0.33 export-capable venv) without editing this script.
PY="${PY:-/home/zcemml1/venvs/medtronic-qat/bin/python}"

RUN_NAME="${RUN_NAME:?RUN_NAME is required (e.g. qat_smoke)}"
DEVICE="${DEVICE:?DEVICE is required -- no default, same discipline as the benchmarks}"
EPOCHS="${EPOCHS:-1}"

SCRATCH="/tmp/zcemml1_qat"                 # local, off-quota, non-durable
NFS_OUT="$REPO/runs_qat/$RUN_NAME"         # durable, small: final artifacts only
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_qat.py}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

cd "$REPO" || exit 1
mkdir -p "$SCRATCH" "$NFS_OUT"

echo "[$(stamp)] QAT run '$RUN_NAME'  device=$DEVICE  epochs=$EPOCHS"
echo "[$(stamp)] scratch (churn):  $SCRATCH"
echo "[$(stamp)] durable (final):  $NFS_OUT"

# --- pre-flight: both filesystems must have room -------------------------------
quota_free_mb=$(quota -s 2>/dev/null | grep -A1 "vol_home_2/zcemml1" | tail -1 \
                | awk '{print int($3)-int($1)}')
tmp_free_mb=$(df -m /tmp | tail -1 | awk '{print $4}')
echo "[$(stamp)] NFS quota free: ${quota_free_mb:-?} MB   /tmp free: ${tmp_free_mb} MB"

if [ "${tmp_free_mb:-0}" -lt 5000 ]; then
    echo "[$(stamp)] ABORT: /tmp has <5 GB free; checkpoint churn needs room."
    exit 1
fi
if [ "${quota_free_mb:-0}" -lt 1000 ]; then
    echo "[$(stamp)] ABORT: NFS quota has <1 GB free; copy-back would fail."
    exit 1
fi

# --- copy-back runs even if training dies --------------------------------------
# EXIT trap, not a trailing command: a crashed or interrupted run should still
# preserve whatever checkpoints it managed to write.
copy_back() {
    rc=$?
    src="$SCRATCH/$RUN_NAME"
    echo "[$(stamp)] copy-back (exit code $rc)"
    if [ -d "$src" ]; then
        # The DURABLE QAT artifact is the modelopt state file, not best.pt:
        # save=False means Ultralytics writes no best.pt (its torch.save cannot
        # pickle modelopt's dynamic classes). qat_modelopt_state.pt carries the
        # recipe + trained weights and is reloadable via mto.restore onto the
        # committed V1 baseline architecture. The .pt per-epoch churn never
        # existed, so nothing large is left on scratch.
        for f in qat_modelopt_state.pt qat_modelopt_state_best.pt qat_provenance.json results.csv args.yaml; do
            [ -f "$src/$f" ] && cp -p "$src/$f" "$NFS_OUT/" \
                && echo "[$(stamp)]   saved $f ($(du -h "$src/$f" | cut -f1))"
        done
    else
        echo "[$(stamp)]   no run dir at $src -- nothing to preserve"
    fi
    echo "[$(stamp)] durable artifacts now in $NFS_OUT:"
    ls -la "$NFS_OUT" 2>/dev/null | tail -n +2 | awk '{print "             "$5"  "$9}'
    exit $rc
}
trap copy_back EXIT

# --- train ---------------------------------------------------------------------
DEVICE="$DEVICE" EPOCHS="$EPOCHS" RUN_NAME="$RUN_NAME" PROJECT="$SCRATCH" \
    "$PY" -u "$TRAIN_SCRIPT"

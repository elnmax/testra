#!/usr/bin/env bash
# Test an arbitrary checkpoint via tools/test_net_optim.py (stream).
#
# Usage:
#   ./scripts/test_checkpoint.sh tvseries
#   ./scripts/test_checkpoint.sh thumos
#   ./scripts/test_checkpoint.sh thumos /path/to/checkpoint.pth
#
# GPU: GPU variable (defaults to 0).
# DATA_ROOT: dataset root (defaults to datasets/TVSeries or datasets/THUMOS).
# CHECKPOINT: path to a .pth file (defaults to the matching file in checkpoints).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate the project venv first (source .../bin/activate)." >&2
  exit 1
fi
PYTHON="${VIRTUAL_ENV}/bin/python"

DATASET="${1:?Specify tvseries or thumos}"
GPU="${GPU:-0}"

case "${DATASET}" in
  tvseries)
    CONFIG="configs/TVSeries/TESTRA/testra_long_512_work_8_kinetics_1x_box.yaml"
    DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets/TVSeries}"
    DEFAULT_CHECKPOINT="${ROOT_DIR}/checkpoints/tvseries_testra.pth"
    ;;
  thumos)
    CONFIG="configs/THUMOS/TESTRA/testra_long_512_work_8_kinetics_1x_box.yaml"
    DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets/THUMOS14}"
    DEFAULT_CHECKPOINT="${ROOT_DIR}/checkpoints/thumos_testra.pth"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}. Use tvseries or thumos." >&2
    exit 1
    ;;
esac

CHECKPOINT="${2:-${CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint file not found: ${CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON}" tools/test_net_optim.py \
  --config_file "${CONFIG}" \
  --gpu "${GPU}" \
  DATA.DATA_ROOT "${DATA_ROOT}" \
  INPUT.MODALITY motion \
  INPUT.MOTION_FEATURE features/repr_x3d_test \
  MODEL.CHECKPOINT "${CHECKPOINT}" \
  MODEL.LSTR.INFERENCE_MODE stream

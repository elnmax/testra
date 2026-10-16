#!/usr/bin/env bash
# Разовый бейслайн TeSTra на THUMOS: 3 сида, train_net.py, тест epoch через test_net_optim.py.
#
# RUN_MODE: both (default) | train | test
#   both  — обучение и test
#   train — только обучение
#   test  — только test по готовым epoch-${LAST_EPOCH}.pth
#
# MODALITY: twostream (default) | visual | motion | threestream
#   twostream  — visual + motion (INPUT.MODALITY twostream в yaml/cli)
#   visual     — только VISUAL_FEATURE
#   motion     — только MOTION_FEATURE
#
# Примеры:
#   MODALITY=motion MOTION_FEATURE=features/x3d_xs_embedding_2048 \
#       SESSION_PREFIX=baseline_x3d_motion ./scripts/run_baseline_thumos_3seeds.sh
#   MODALITY=visual VISUAL_FEATURE=features/mnv2_embedding_1280 \
#       SESSION_PREFIX=baseline_mnv2_visual ./scripts/run_baseline_thumos_3seeds.sh
#   VISUAL_FEATURE=features/mnv2_embedding_1280 \
#       MOTION_FEATURE=features/x3d_xs_embedding_2048 \
#       SESSION_PREFIX=baseline_mnv2_x3d ./scripts/run_baseline_thumos_3seeds.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

CFG="${CFG:-configs/THUMOS/TESTRA/testra_long_512_work_8_kinetics_1x_box.yaml}"
DATA_ROOT="${DATA_ROOT:-/workspace/QAT-neuro-transformer/datasets/THUMOS14}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-66706 12345 3225}"
LAST_EPOCH="${LAST_EPOCH:-25}"
PYTHON="${PYTHON:-python}"
SESSION_PREFIX="${SESSION_PREFIX:-baseline}"
RUN_MODE="${RUN_MODE:-both}"
INFERENCE_MODE="${INFERENCE_MODE:-stream}"
MODALITY="${MODALITY:-twostream}"

VISUAL_FEATURE="${VISUAL_FEATURE:-features/testra_features/rgb_kinetics_resnet50}"
MOTION_FEATURE="${MOTION_FEATURE:-features/testra_features/flow_kinetics_bninception}"
OBJECT_FEATURE="${OBJECT_FEATURE:-features/testra_features/flow_kinetics_bninception}"

CKPT_ROOT="checkpoints/THUMOS/TESTRA/testra_long_512_work_8_kinetics_1x_box"

COMMON_OPTS=(
  --config_file "${CFG}"
  --gpu "${GPU}"
  DATA.DATA_ROOT "${DATA_ROOT}"
  INPUT.MODALITY "${MODALITY}"
  INPUT.VISUAL_FEATURE "${VISUAL_FEATURE}"
  INPUT.MOTION_FEATURE "${MOTION_FEATURE}"
  INPUT.OBJECT_FEATURE "${OBJECT_FEATURE}"
)

run_dir() {
  local seed="$1"
  echo "${CKPT_ROOT}/${SESSION_PREFIX}_s${seed}"
}

read_map_from_logs() {
  local base_path="$1"
  local optim_log="${base_path}.optim.txt"
  local plain_log="${base_path}.txt"
  local map_line=""

  if [[ -f "${optim_log}" ]]; then
    map_line="$(grep "perframe mAP" "${optim_log}" | tail -n 1 || true)"
  fi
  if [[ -z "${map_line}" && -f "${plain_log}" ]]; then
    map_line="$(grep "perframe mAP" "${plain_log}" | tail -n 1 || true)"
  fi

  if [[ -n "${map_line}" ]]; then
    echo "${map_line}"
    return
  fi

  if [[ -f "${optim_log}" ]]; then
    echo "не найдено в ${optim_log}"
  elif [[ -f "${plain_log}" ]]; then
    echo "не найдено в ${plain_log}"
  else
    echo "не найдено: ${optim_log} или ${plain_log}"
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Нет каталога: ${path}" >&2
    exit 1
  fi
}

validate_run_mode() {
  case "${RUN_MODE}" in
    both|train|test) ;;
    *)
      echo "RUN_MODE должен быть both, train или test, получено: ${RUN_MODE}" >&2
      exit 1
      ;;
  esac
}

validate_modality() {
  case "${MODALITY}" in
    twostream|threestream|visual|motion|object|visual+motion|visual+motion+object) ;;
    *)
      echo "MODALITY должен быть twostream, visual, motion, threestream и т.д., получено: ${MODALITY}" >&2
      exit 1
      ;;
  esac
}

validate_run_mode
validate_modality
require_dir "${DATA_ROOT}/target_perframe"
require_dir "${DATA_ROOT}/${VISUAL_FEATURE}"
require_dir "${DATA_ROOT}/${MOTION_FEATURE}"
require_dir "${DATA_ROOT}/${OBJECT_FEATURE}"

echo "=== TeSTra baseline | GPU=${GPU} | seeds=${SEEDS} | epoch=${LAST_EPOCH} | RUN_MODE=${RUN_MODE} | MODALITY=${MODALITY} ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "VISUAL_FEATURE=${VISUAL_FEATURE}"
echo "MOTION_FEATURE=${MOTION_FEATURE}"

for seed in ${SEEDS}; do
  session="${SESSION_PREFIX}_s${seed}"
  ckpt="$(run_dir "${seed}")/epoch-${LAST_EPOCH}.pth"

  if [[ "${RUN_MODE}" != "test" ]]; then
    echo
    echo "=== TRAIN seed=${seed} SESSION=${session} ==="
    "${PYTHON}" tools/train_net.py \
      "${COMMON_OPTS[@]}" \
      SEED "${seed}" \
      SESSION "${session}"
  fi

  if [[ "${RUN_MODE}" != "train" ]]; then
    if [[ ! -f "${ckpt}" ]]; then
      echo "Нет чекпоинта: ${ckpt}" >&2
      exit 1
    fi

    echo
    echo "=== TEST (optim) seed=${seed} ckpt=${ckpt} ==="
    "${PYTHON}" tools/test_net_optim.py \
      "${COMMON_OPTS[@]}" \
      SESSION "${session}" \
      MODEL.CHECKPOINT "${ckpt}" \
      MODEL.LSTR.INFERENCE_MODE "${INFERENCE_MODE}"
  fi
done

if [[ "${RUN_MODE}" != "train" ]]; then
  echo
  echo "=== mAP последней эпохи (optim) ==="
  for seed in ${SEEDS}; do
    log_base="$(run_dir "${seed}")/epoch-${LAST_EPOCH}"
    map="$(read_map_from_logs "${log_base}")"
    printf "seed %s: %s\n" "${seed}" "${map}"
  done
fi

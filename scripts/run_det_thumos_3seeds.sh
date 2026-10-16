#!/usr/bin/env bash
# Детерминированный бейслайн TeSTra на THUMOS: 3 пары seed_model/seed_data,
# train_net_det.py, тест best.pth через test_net_optim.py.
#
# RUN_MODE: both (default) | train | test
#   both  — обучение и test best
#   train — только обучение
#   test  — только test по готовому best.pth
#
# MODALITY: twostream (default) | visual | motion | threestream
#   twostream  — visual + motion
#   visual     — только VISUAL_FEATURE
#   motion     — только MOTION_FEATURE
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

CFG="${CFG:-configs/THUMOS/TESTRA/testra_long_512_work_8_kinetics_1x_box.yaml}"
DATA_ROOT="${DATA_ROOT:-/workspace/QAT-neuro-transformer/datasets/THUMOS14}"
GPU="${GPU:-0}"
SEED_MODELS="${SEED_MODELS:-66706 12345 3225}"
SEED_DATA="${SEED_DATA:-3357 67890 12}"
PYTHON="${PYTHON:-python}"
SESSION_PREFIX="${SESSION_PREFIX:-det}"
RUN_MODE="${RUN_MODE:-both}"
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
  local seed_model="$1"
  local seed_data="$2"
  echo "${CKPT_ROOT}/${SESSION_PREFIX}_s${seed_model}_d${seed_data}"
}

read_ckpt_epoch() {
  local ckpt_path="$1"
  "${PYTHON}" -c "
import torch
ckpt = torch.load('${ckpt_path}', map_location='cpu', weights_only=False)
print(int(ckpt['epoch']))
"
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

validate_seed_lists() {
  local -a model_seeds=(${SEED_MODELS})
  local -a data_seeds=(${SEED_DATA})

  if [[ ${#model_seeds[@]} -ne ${#data_seeds[@]} ]]; then
    echo "SEED_MODELS и SEED_DATA должны иметь одинаковую длину" >&2
    echo "SEED_MODELS (${#model_seeds[@]}): ${SEED_MODELS}" >&2
    echo "SEED_DATA (${#data_seeds[@]}): ${SEED_DATA}" >&2
    exit 1
  fi
}

validate_run_mode
validate_modality
require_dir "${DATA_ROOT}/target_perframe"
require_dir "${DATA_ROOT}/${VISUAL_FEATURE}"
require_dir "${DATA_ROOT}/${MOTION_FEATURE}"
require_dir "${DATA_ROOT}/${OBJECT_FEATURE}"
validate_seed_lists

echo "=== TeSTra det | GPU=${GPU} | model=${SEED_MODELS} | data=${SEED_DATA} | RUN_MODE=${RUN_MODE} | MODALITY=${MODALITY} ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "VISUAL_FEATURE=${VISUAL_FEATURE}"
echo "MOTION_FEATURE=${MOTION_FEATURE}"

read -r -a model_seeds <<< "${SEED_MODELS}"
read -r -a data_seeds <<< "${SEED_DATA}"

declare -A best_epochs

for idx in "${!model_seeds[@]}"; do
  seed_model="${model_seeds[${idx}]}"
  seed_data="${data_seeds[${idx}]}"
  session="${SESSION_PREFIX}_s${seed_model}_d${seed_data}"
  run_path="$(run_dir "${seed_model}" "${seed_data}")"
  run_key="${seed_model}_${seed_data}"

  if [[ "${RUN_MODE}" != "test" ]]; then
    echo
    echo "=== TRAIN seed_model=${seed_model} seed_data=${seed_data} SESSION=${session} ==="
    "${PYTHON}" tools/train_net_det.py \
      "${COMMON_OPTS[@]}" \
      SEED "${seed_model}" \
      SEED_DATA "${seed_data}" \
      SESSION "${session}"
  fi

  if [[ "${RUN_MODE}" != "train" ]]; then
    best_ckpt="${run_path}/best.pth"
    if [[ ! -f "${best_ckpt}" ]]; then
      echo "Нет чекпоинта best.pth в ${run_path}" >&2
      exit 1
    fi

    best_epochs["${run_key}"]="$(read_ckpt_epoch "${best_ckpt}")"
    echo "best epoch=${best_epochs[${run_key}]}"

    echo
    echo "=== TEST (optim) seed_model=${seed_model} seed_data=${seed_data} best epoch=${best_epochs[${run_key}]} ckpt=${best_ckpt} ==="
    "${PYTHON}" tools/test_net_optim.py \
      "${COMMON_OPTS[@]}" \
      SESSION "${session}" \
      MODEL.CHECKPOINT "${best_ckpt}"
  fi
done

if [[ "${RUN_MODE}" != "train" ]]; then
  echo
  echo "=== mAP best ==="
  for idx in "${!model_seeds[@]}"; do
    seed_model="${model_seeds[${idx}]}"
    seed_data="${data_seeds[${idx}]}"
    run_path="$(run_dir "${seed_model}" "${seed_data}")"
    run_key="${seed_model}_${seed_data}"

    if [[ -z "${best_epochs[${run_key}]:-}" ]]; then
      best_epochs["${run_key}"]="$(read_ckpt_epoch "${run_path}/best.pth")"
    fi

    best_log="${run_path}/best"
    best_map="$(read_map_from_logs "${best_log}")"
    printf "seed_model %s seed_data %s best_epoch %s: %s\n" \
      "${seed_model}" "${seed_data}" "${best_epochs[${run_key}]}" \
      "${best_map}"
  done
fi

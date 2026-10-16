#!/usr/bin/env bash
# TeSTra baseline на THUMOS14: только TEST через tools/test_net_optim.py (stream).
#
# Основное отличие от run_baseline_thumos_3seeds.sh:
#   - используем тест-движок с оптимизациями: tools/test_net_optim.py
#   - поддерживаем multi-GPU через параметр сколько GPU использовать
#
# GPU selection:
#   - GPU_LIST="0,1,2" (приоритет)
#   - иначе GPU_START=0 и NGPUS=1 => "0" / "0,1" / ...
#
# RUN: для каждого seed запускается тест epoch-${LAST_EPOCH}.pth.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CFG="${CFG:-configs/THUMOS/TESTRA/testra_long_512_work_8_kinetics_1x_box.yaml}"
DATA_ROOT="${DATA_ROOT:-/workspace/QAT-neuro-transformer/datasets/THUMOS14}"

# GPU selection
NGPUS="${NGPUS:-1}"
GPU_START="${GPU_START:-0}"
GPU_LIST="${GPU_LIST:-}"
GPU="${GPU:-}" # backward compat

SEEDS="${SEEDS:-66706 12345 3225}"
LAST_EPOCH="${LAST_EPOCH:-25}"

PYTHON="${PYTHON:-python}"
SESSION_PREFIX="${SESSION_PREFIX:-baseline}"
INFERENCE_MODE="${INFERENCE_MODE:-stream}"
MODALITY="${MODALITY:-twostream}"

VISUAL_FEATURE="${VISUAL_FEATURE:-features/testra_features/rgb_kinetics_resnet50}"
MOTION_FEATURE="${MOTION_FEATURE:-features/testra_features/flow_kinetics_bninception}"
OBJECT_FEATURE="${OBJECT_FEATURE:-features/testra_features/flow_kinetics_bninception}"

CKPT_ROOT="checkpoints/THUMOS/TESTRA/testra_long_512_work_8_kinetics_1x_box"

gen_gpu_list() {
  local start="$1"
  local n="$2"
  local out=""
  local i
  for ((i = 0; i < n; i++)); do
    local gid=$((start + i))
    if [[ -z "${out}" ]]; then
      out="${gid}"
    else
      out="${out},${gid}"
    fi
  done
  echo "${out}"
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

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Нет каталога: ${path}" >&2
    exit 1
  fi
}

compute_gpu_ids() {
  if [[ -n "${GPU_LIST}" ]]; then
    echo "${GPU_LIST}"
    return
  fi

  # backward compatibility: if user passed GPU="0" etc.
  if [[ -n "${GPU}" ]]; then
    echo "${GPU}"
    return
  fi

  gen_gpu_list "${GPU_START}" "${NGPUS}"
}

GPU_IDS="$(compute_gpu_ids)"

COMMON_OPTS=(
  --config_file "${CFG}"
  --gpu "${GPU_IDS}"
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

validate_modality

require_dir "${DATA_ROOT}/target_perframe"

# Require only feature folders that are actually used by the selected modality.
case "${MODALITY}" in
  twostream|visual+motion)
    require_dir "${DATA_ROOT}/${VISUAL_FEATURE}"
    require_dir "${DATA_ROOT}/${MOTION_FEATURE}"
    ;;
  visual)
    require_dir "${DATA_ROOT}/${VISUAL_FEATURE}"
    ;;
  motion)
    require_dir "${DATA_ROOT}/${MOTION_FEATURE}"
    ;;
  threestream|visual+motion+object)
    require_dir "${DATA_ROOT}/${VISUAL_FEATURE}"
    require_dir "${DATA_ROOT}/${MOTION_FEATURE}"
    require_dir "${DATA_ROOT}/${OBJECT_FEATURE}"
    ;;
  object)
    require_dir "${DATA_ROOT}/${OBJECT_FEATURE}"
    ;;
esac

echo "=== TeSTra baseline | optim only | GPU_IDS=${GPU_IDS} | seeds=${SEEDS} | epoch=${LAST_EPOCH} | MODALITY=${MODALITY} ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "VISUAL_FEATURE=${VISUAL_FEATURE}"
echo "MOTION_FEATURE=${MOTION_FEATURE}"

for seed in ${SEEDS}; do
  session="${SESSION_PREFIX}_s${seed}"
  run_path="$(run_dir "${seed}")"
  ckpt="${run_path}/epoch-${LAST_EPOCH}.pth"

  if [[ ! -f "${ckpt}" ]]; then
    echo "Нет чекпоинта: ${ckpt}" >&2
    exit 1
  fi

  epoch="$(read_ckpt_epoch "${ckpt}")"
  echo
  echo "=== TEST optim seed=${seed} SESSION=${session} ckpt=${ckpt} (epoch=${epoch}) ==="

  "${PYTHON}" tools/test_net_optim.py \
    "${COMMON_OPTS[@]}" \
    SESSION "${session}" \
    MODEL.CHECKPOINT "${ckpt}" \
    MODEL.LSTR.INFERENCE_MODE "${INFERENCE_MODE}"
done

echo
echo "=== mAP последней эпохи (optim) ==="
for seed in ${SEEDS}; do
  log_base="$(run_dir "${seed}")/epoch-${LAST_EPOCH}"
  map="$(read_map_from_logs "${log_base}")"
  printf "seed %s: %s\n" "${seed}" "${map}"
done


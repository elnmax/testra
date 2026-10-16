#!/usr/bin/env bash
# TeSTra det на THUMOS14: только TEST через tools/test_net_optim.py (stream).
#
# Проходит по парам SEED_MODELS / SEED_DATA (как run_det_thumos_3seeds.sh).
# По умолчанию тестирует оба чекпоинта:
#   - last.pth
#   - best.pth
#
# CHECKPOINT_MODE:
#   both (default) — last и best
#   last           — только last.pth
#   best           — только best.pth
#
# GPU selection:
#   - GPU_LIST="0,1,2" (приоритет)
#   - иначе GPU_START=0 и NGPUS=1 => "0" / "0,1" / ...
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CFG="${CFG:-configs/THUMOS/TESTRA/testra_long_512_work_8_kinetics_1x_box.yaml}"
DATA_ROOT="${DATA_ROOT:-/workspace/QAT-neuro-transformer/datasets/THUMOS14}"

NGPUS="${NGPUS:-1}"
GPU_START="${GPU_START:-0}"
GPU_LIST="${GPU_LIST:-}"
GPU="${GPU:-}"

SEED_MODELS="${SEED_MODELS:-66706 12345 3225}"
SEED_DATA="${SEED_DATA:-3357 67890 12}"

PYTHON="${PYTHON:-python}"
SESSION_PREFIX="${SESSION_PREFIX:-det}"
INFERENCE_MODE="${INFERENCE_MODE:-stream}"
MODALITY="${MODALITY:-twostream}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-both}"

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

validate_checkpoint_mode() {
  case "${CHECKPOINT_MODE}" in
    both|last|best) ;;
    *)
      echo "CHECKPOINT_MODE должен быть both, last или best, получено: ${CHECKPOINT_MODE}" >&2
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

  if [[ -n "${GPU}" ]]; then
    echo "${GPU}"
    return
  fi

  gen_gpu_list "${GPU_START}" "${NGPUS}"
}

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

run_test_checkpoint() {
  local seed_model="$1"
  local seed_data="$2"
  local label="$3"
  local ckpt_path="$4"
  local epoch="$5"
  local session="${SESSION_PREFIX}_s${seed_model}_d${seed_data}"

  echo
  echo "=== TEST optim seed_model=${seed_model} seed_data=${seed_data} SESSION=${session} label=${label} ckpt=${ckpt_path} (epoch=${epoch}) ==="

  "${PYTHON}" tools/test_net_optim.py \
    "${COMMON_OPTS[@]}" \
    SESSION "${session}" \
    MODEL.CHECKPOINT "${ckpt_path}" \
    MODEL.LSTR.INFERENCE_MODE "${INFERENCE_MODE}"
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

validate_modality
validate_checkpoint_mode
validate_seed_lists

require_dir "${DATA_ROOT}/target_perframe"

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

echo "=== TeSTra det | optim test | GPU_IDS=${GPU_IDS} | model=${SEED_MODELS} | data=${SEED_DATA} | CHECKPOINT_MODE=${CHECKPOINT_MODE} | MODALITY=${MODALITY} ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "VISUAL_FEATURE=${VISUAL_FEATURE}"
echo "MOTION_FEATURE=${MOTION_FEATURE}"

read -r -a model_seeds <<< "${SEED_MODELS}"
read -r -a data_seeds <<< "${SEED_DATA}"

declare -A last_epochs
declare -A best_epochs

for idx in "${!model_seeds[@]}"; do
  seed_model="${model_seeds[${idx}]}"
  seed_data="${data_seeds[${idx}]}"
  run_path="$(run_dir "${seed_model}" "${seed_data}")"
  run_key="${seed_model}_${seed_data}"

  last_ckpt="${run_path}/last.pth"
  best_ckpt="${run_path}/best.pth"

  if [[ "${CHECKPOINT_MODE}" == "both" || "${CHECKPOINT_MODE}" == "last" ]]; then
    if [[ ! -f "${last_ckpt}" ]]; then
      echo "Нет чекпоинта: ${last_ckpt}" >&2
      exit 1
    fi
    last_epochs["${run_key}"]="$(read_ckpt_epoch "${last_ckpt}")"
    run_test_checkpoint \
      "${seed_model}" "${seed_data}" "last" \
      "${last_ckpt}" "${last_epochs[${run_key}]}"
  fi

  if [[ "${CHECKPOINT_MODE}" == "both" || "${CHECKPOINT_MODE}" == "best" ]]; then
    if [[ ! -f "${best_ckpt}" ]]; then
      echo "Нет чекпоинта: ${best_ckpt}" >&2
      exit 1
    fi
    best_epochs["${run_key}"]="$(read_ckpt_epoch "${best_ckpt}")"
    run_test_checkpoint \
      "${seed_model}" "${seed_data}" "best" \
      "${best_ckpt}" "${best_epochs[${run_key}]}"
  fi
done

if [[ "${CHECKPOINT_MODE}" == "both" || "${CHECKPOINT_MODE}" == "last" ]]; then
  echo
  echo "=== mAP last (optim) ==="
  for idx in "${!model_seeds[@]}"; do
    seed_model="${model_seeds[${idx}]}"
    seed_data="${data_seeds[${idx}]}"
    run_path="$(run_dir "${seed_model}" "${seed_data}")"
    run_key="${seed_model}_${seed_data}"

    if [[ -z "${last_epochs[${run_key}]:-}" ]]; then
      last_epochs["${run_key}"]="$(read_ckpt_epoch "${run_path}/last.pth")"
    fi

    map_line="$(read_map_from_logs "${run_path}/last")"
    printf "seed_model %s seed_data %s last_epoch %s: %s\n" \
      "${seed_model}" "${seed_data}" "${last_epochs[${run_key}]}" "${map_line}"
  done
fi

if [[ "${CHECKPOINT_MODE}" == "both" || "${CHECKPOINT_MODE}" == "best" ]]; then
  echo
  echo "=== mAP best (optim) ==="
  for idx in "${!model_seeds[@]}"; do
    seed_model="${model_seeds[${idx}]}"
    seed_data="${data_seeds[${idx}]}"
    run_path="$(run_dir "${seed_model}" "${seed_data}")"
    run_key="${seed_model}_${seed_data}"

    if [[ -z "${best_epochs[${run_key}]:-}" ]]; then
      best_epochs["${run_key}"]="$(read_ckpt_epoch "${run_path}/best.pth")"
    fi

    map_line="$(read_map_from_logs "${run_path}/best")"
    printf "seed_model %s seed_data %s best_epoch %s: %s\n" \
      "${seed_model}" "${seed_data}" "${best_epochs[${run_key}]}" "${map_line}"
  done
fi

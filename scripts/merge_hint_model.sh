#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/ly/agentpo:${PYTHONPATH:-}
export HF_HOME=/mnt/huawei/leiy/hug
export HF_HUB_CACHE=/mnt/huawei/leiy/hug/hub
export HF_ASSETS_CACHE=/mnt/huawei/leiy/hug/assets
export HF_TOKEN_PATH=/mnt/huawei/leiy/hug/token

REPO_ROOT="/home/ly/agentpo"
CKPT_ROOT="/mnt/huawei/leiy/checkpoints/agentpo"
PROJECT_NAME="AgentPO"
EXP_NAME="${EXP_NAME:-assistant_Qwen2.5-3B_Llama-3.2-3B_500}"
STEP="${STEP:-last}"

CKPT_DIR="${CKPT_ROOT}/${PROJECT_NAME}/${EXP_NAME}"
LOCAL_DIR="${LOCAL_DIR:-${CKPT_DIR}/${STEP}/actor}"
TARGET_DIR="${TARGET_DIR:-${CKPT_DIR}/merged_hf_model_${STEP}}"

if [[ ! -d "${LOCAL_DIR}" ]]; then
    echo "Checkpoint actor dir not found: ${LOCAL_DIR}" >&2
    exit 1
fi

echo "Merging FSDP checkpoint:"
echo "  local_dir : ${LOCAL_DIR}"
echo "  target_dir: ${TARGET_DIR}"

cd "${REPO_ROOT}"

python verl/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}"

echo "Merged HF model saved to: ${TARGET_DIR}"

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/ly/agentpo/agentpo/evaluation:/home/ly/agentpo:${PYTHONPATH:-}
export HF_HOME=/mnt/huawei/leiy/hug
export HF_HUB_CACHE=/mnt/huawei/leiy/hug/hub
export HF_ASSETS_CACHE=/mnt/huawei/leiy/hug/assets
export HF_TOKEN_PATH=/mnt/huawei/leiy/hug/token
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="/home/ly/agentpo"
CKPT_ROOT="/mnt/huawei/leiy/checkpoints/agentpo"
PROJECT_NAME="AgentPO"

EXP_NAME="${EXP_NAME:-baseline_Llama-3.2-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_ROOT}/${PROJECT_NAME}/${EXP_NAME}/eval_api}"
METRICS_DIR="${METRICS_DIR:-/home/ly/agentpo/metric}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot-extra}"
DATA_NAMES="${DATA_NAMES:-aime24,math500,olympiadbench,minerva_math,amc23}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/agentpo/evaluation/data}"
NUM_TEST_SAMPLE="${NUM_TEST_SAMPLE:--1}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-0}"
START="${START:-0}"
END="${END:--1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_K="${MAX_K:-1}"
API_BATCH_SIZE="${API_BATCH_SIZE:-64}"
USE_PASS_K="${USE_PASS_K:-False}"
ACTOR_MODEL="${ACTOR_MODEL:-Llama-3.2-3B}"

echo "Running Llama API-only evaluation:"
echo "  actor model: ${ACTOR_MODEL} API"
echo "  data       : ${DATA_NAMES}"
echo "  samples    : ${NUM_TEST_SAMPLE}"
echo "  max_tokens : ${MAX_TOKENS}"
echo "  n_sampling : ${MAX_K}"
echo "  api_batch  : ${API_BATCH_SIZE}"
echo "  output_dir : ${OUTPUT_DIR}"
echo "  metrics_dir: ${METRICS_DIR}"

cd "${REPO_ROOT}"

python -u agentpo/evaluation/math_eval_promptpo.py \
    --api_only_actor \
    --model_name_or_path "${ACTOR_MODEL}" \
    --exp_name "${EXP_NAME}" \
    --max_tokens_per_call "${MAX_TOKENS}" \
    --data_names "${DATA_NAMES}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --metrics_dir "${METRICS_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --actor_model "${ACTOR_MODEL}" \
    --cooperation_mode base \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --n_sampling "${MAX_K}" \
    --api_batch_size "${API_BATCH_SIZE}" \
    --start "${START}" \
    --end "${END}" \
    --save_outputs True \
    --overwrite "${OVERWRITE:-True}" \
    --use_pass_k "${USE_PASS_K}"

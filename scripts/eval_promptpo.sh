#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/ly/agentpo/agentpo/evaluation:/home/ly/agentpo:${PYTHONPATH:-}
export HF_HOME=/mnt/huawei/leiy/hug
export HF_HUB_CACHE=/mnt/huawei/leiy/hug/hub
export HF_ASSETS_CACHE=/mnt/huawei/leiy/hug/assets
export HF_TOKEN_PATH=/mnt/huawei/leiy/hug/token
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="/home/ly/agentpo"
CKPT_ROOT="/mnt/huawei/leiy/checkpoints/agentpo"
PROJECT_NAME="AgentPO"
EXP_NAME="${EXP_NAME:-assistant_Qwen2.5-3B_Llama-3.2-3B_500}"
STEP="${STEP:-last}"

MODEL_PATH="${MODEL_PATH:-${CKPT_ROOT}/${PROJECT_NAME}/${EXP_NAME}/merged_hf_model_${STEP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_ROOT}/${PROJECT_NAME}/${EXP_NAME}/eval_promptpo_${STEP}}"
METRICS_DIR="${METRICS_DIR:-/home/ly/agentpo/metric}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot-extra}"
DATA_NAMES="${DATA_NAMES:-math8k}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/agentpo/evaluation/data}"
NUM_TEST_SAMPLE="${NUM_TEST_SAMPLE:-50}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-0}"
START="${START:-0}"
END="${END:--1}"
TEMPERATURE="${TEMPERATURE:-1.5}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_K="${MAX_K:-1}"
USE_PASS_K="${USE_PASS_K:-False}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Merged HF model dir not found: ${MODEL_PATH}" >&2
    echo "Run scripts/merge_hint_model.sh first, or set MODEL_PATH=/path/to/merged_hf_model." >&2
    exit 1
fi

echo "Running PromptPO evaluation:"
echo "  hint model : ${MODEL_PATH}"
echo "  actor model: Llama-3.2-3B API at http://127.0.0.1:10005/v1"
echo "  data       : ${DATA_NAMES}"
echo "  samples    : ${NUM_TEST_SAMPLE}"
echo "  max_tokens : ${MAX_TOKENS}"
echo "  n_sampling : ${MAX_K}"
echo "  output_dir : ${OUTPUT_DIR}"
echo "  metrics_dir: ${METRICS_DIR}"

cd "${REPO_ROOT}"

python -u agentpo/evaluation/math_eval_promptpo.py \
    --model_name_or_path "${MODEL_PATH}" \
    --exp_name "${EXP_NAME}" \
    --max_tokens_per_call "${MAX_TOKENS}" \
    --data_names "${DATA_NAMES}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --metrics_dir "${METRICS_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --tokenizer_mode "${TOKENIZER_MODE:-slow}" \
    --actor_model "${ACTOR_MODEL:-Llama-3.2-3B}" \
    --cooperation_mode "${COOPERATION_MODE:-assistant}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --n_sampling "${MAX_K}" \
    --start "${START}" \
    --end "${END}" \
    --use_vllm True \
    --save_outputs True \
    --overwrite "${OVERWRITE:-True}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --use_pass_k "${USE_PASS_K}"

#!/usr/bin/env bash
set -e

export PYTHONPATH=/home/ly/agentpo/agentpo/evaluation:/home/ly/agentpo:$PYTHONPATH
export HF_HOME=/mnt/huawei/leiy/hug
export HF_HUB_CACHE=/mnt/huawei/leiy/hug/hub
export HF_ASSETS_CACHE=/mnt/huawei/leiy/hug/assets
export HF_TOKEN_PATH=/mnt/huawei/leiy/hug/token
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CKPT_DIR="/mnt/huawei/leiy/checkpoints/agentpo/AgentPO/smoke_Qwen2.5-3B_Llama-3.2-3B"
CHECKPOINT_PATH="${CKPT_DIR}/last/actor"
MERGE_MODEL_PATH="${CKPT_DIR}/merged_hf_model_last"
OUTPUT_DIR="${CKPT_DIR}/eval_smoke"

python verl/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "${CHECKPOINT_PATH}" \
    --target_dir "${MERGE_MODEL_PATH}"

TOKENIZERS_PARALLELISM=false \
python -u agentpo/evaluation/math_eval.py \
    --model_name_or_path "${MERGE_MODEL_PATH}" \
    --max_tokens_per_call 512 \
    --data_names math8k \
    --data_dir "/home/ly/agentpo/agentpo/evaluation/data" \
    --output_dir "${OUTPUT_DIR}" \
    --split test \
    --prompt_type qwen25-math-cot-extra \
    --num_test_sample 2 \
    --tokenizer_mode slow \
    --seed 0 \
    --temperature 0.7 \
    --top_p 1.0 \
    --top_k -1 \
    --n_sampling 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --gpu_memory_utilization 0.6 \
    --use_pass_k False

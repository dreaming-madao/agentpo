set -ex

export CUDA_VISIBLE_DEVICES=5
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

LOSS_TYPE="dapo_7B_passk_1.5" # focal_1.5B
STEP="last"

# checkpoint_path=/home/sunl/verl_rl/runs/DAPO/DAPO-Qwen2.5-7B/last/actor
# MERGE_MODEL_PATH=/home/sunl/verl_rl/runs/DAPO/DAPO-Qwen2.5-7B/merged_hf_model_${STEP}
MERGE_MODEL_PATH=/home/sunl/verl_rl/runs/EXPO/DAPO-Qwen2.5-7B
# MERGE_MODEL_PATH=/home/sunl/verl_rl/ckpts/Qwen_Qwen2.5-Math-7B

OUTPUT_DIR="runs/${LOSS_TYPE}_${STEP}"

PROMPT_TYPE="qwen25-math-cot-extra"  #  llama3

MAX_K=128
USE_PASS_K=true
DATA_NAME="aime24,math500,olympiadbench,minerva_math,amc23"
# DATA_NAME="aime24,aime25,math500,olympiadbench,minerva_math,amc23"
SPLIT="test"

NUM_TEST_SAMPLE=-1
SEED=0
START=0
END=-1

TEMPERATURE=1.5
TOP_P=1.0
TOP_K=-1
MAX_TOKENS=2048

# TEMPERATURE=0.6
# TOP_P=0.95
# TOP_K=20
# MAX_TOKENS=32000

# merge checkpoints
python ../verl/scripts/model_merger.py merge  \
    --backend fsdp \
    --local_dir  ${checkpoint_path} \
    --target_dir  ${MERGE_MODEL_PATH} \

# evaluate dataset
TOKENIZERS_PARALLELISM=false \
python -u "agentpo/evaluation/math_eval.py" \
    --model_name_or_path ${MERGE_MODEL_PATH} \
    --max_tokens_per_call ${MAX_TOKENS} \
    --data_name ${DATA_NAME} \
    --data_dir "evaluation/data" \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --n_sampling ${MAX_K} \
    --start ${START} \
    --end ${END} \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --gpu_memory_utilization 0.9 \
    --use_pass_k ${USE_PASS_K} \

# rm -rf ${MERGE_MODEL_PATH}
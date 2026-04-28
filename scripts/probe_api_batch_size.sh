#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/ly/agentpo/agentpo/evaluation:/home/ly/agentpo:${PYTHONPATH:-}

REPO_ROOT="/home/ly/agentpo"
METRIC_DIR="/home/ly/agentpo/metric"

ACTOR_MODEL="${ACTOR_MODEL:-Qwen2.5-7B-SiliconFlow}"
EXP_PREFIX="${EXP_PREFIX:-probe_api_batch}"
DATA_NAMES="${DATA_NAMES:-math500}"
NUM_TEST_SAMPLE_MODE="${NUM_TEST_SAMPLE_MODE:-match_batch}"
NUM_TEST_SAMPLE="${NUM_TEST_SAMPLE:-20}"
MAX_K="${MAX_K:-1}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
BATCH_SIZES="${BATCH_SIZES:-8 12 16 24 32 48 64}"
RESULT_FILE="${RESULT_FILE:-${METRIC_DIR}/${EXP_PREFIX}_$(date +%Y%m%d_%H%M%S).txt}"

mkdir -p "${METRIC_DIR}"

echo "API batch size probe" | tee "${RESULT_FILE}"
echo "actor_model=${ACTOR_MODEL}" | tee -a "${RESULT_FILE}"
echo "data_names=${DATA_NAMES}" | tee -a "${RESULT_FILE}"
echo "num_test_sample_mode=${NUM_TEST_SAMPLE_MODE}" | tee -a "${RESULT_FILE}"
echo "num_test_sample_default=${NUM_TEST_SAMPLE}" | tee -a "${RESULT_FILE}"
echo "max_k=${MAX_K}" | tee -a "${RESULT_FILE}"
echo "max_tokens=${MAX_TOKENS}" | tee -a "${RESULT_FILE}"
echo "batch_sizes=${BATCH_SIZES}" | tee -a "${RESULT_FILE}"
echo "" | tee -a "${RESULT_FILE}"

cd "${REPO_ROOT}"

for b in ${BATCH_SIZES}; do
    exp_name="${EXP_PREFIX}_${ACTOR_MODEL//\//_}_b${b}"
    log_file="${METRIC_DIR}/${exp_name}.log"
    if [[ "${NUM_TEST_SAMPLE_MODE}" == "match_batch" ]]; then
        current_num_test_sample="${b}"
    else
        current_num_test_sample="${NUM_TEST_SAMPLE}"
    fi

    echo "==== API_BATCH_SIZE=${b} ====" | tee -a "${RESULT_FILE}"
    echo "num_test_sample=${current_num_test_sample}" | tee -a "${RESULT_FILE}"

    set +e
    PATH=/mnt/huawei/leiy/envs/apo-eva/bin:$PATH \
    EXP_NAME="${exp_name}" \
    ACTOR_MODEL="${ACTOR_MODEL}" \
    DATA_NAMES="${DATA_NAMES}" \
    NUM_TEST_SAMPLE="${current_num_test_sample}" \
    API_BATCH_SIZE="${b}" \
    MAX_K="${MAX_K}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    bash scripts/eval_llama_api.sh > "${log_file}" 2>&1
    status=$?
    set -e

    if [[ ${status} -ne 0 ]]; then
        echo "batch=${b} status=process_failed log=${log_file}" | tee -a "${RESULT_FILE}"
        continue
    fi

    batch_timing="$(rg -o 'Finished batch .*' -N "${log_file}" | tail -n 1 || true)"

    if rg -q "Error processing a completion in batch|APIConnectionError|ConnectError|ReadTimeout|APITimeoutError|incomplete chunked read|\\[unstable\\]" "${log_file}"; then
        echo "batch=${b} status=unstable log=${log_file}" | tee -a "${RESULT_FILE}"
    else
        echo "batch=${b} status=stable log=${log_file}" | tee -a "${RESULT_FILE}"
    fi

    if [[ -n "${batch_timing}" ]]; then
        echo "batch=${b} timing=${batch_timing}" | tee -a "${RESULT_FILE}"
    fi
done

echo "" | tee -a "${RESULT_FILE}"
echo "Saved summary to ${RESULT_FILE}"

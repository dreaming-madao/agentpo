#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/ly/agentpo:${PYTHONPATH:-}
export HF_HOME=/mnt/huawei/leiy/hug
export HF_HUB_CACHE=/mnt/huawei/leiy/hug/hub
export HF_ASSETS_CACHE=/mnt/huawei/leiy/hug/assets
export HF_TOKEN_PATH=/mnt/huawei/leiy/hug/token
export VLLM_USE_V1=0

# Use GPUs for the trainable collaborator. Keep the vLLM actor server on its
# own CUDA_VISIBLE_DEVICES in the terminal where you launched vllm serve.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

HOME_DIR="/home/ly/agentpo"
CKPT_ROOT="/mnt/huawei/leiy/checkpoints/agentpo"
PROJECT_NAME="AgentPO"
EXP_NAME="${EXP_NAME:-mad_fast_Qwen2.5-3B_Llama-3.2-3B_500}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
ACTOR_MODEL="${ACTOR_MODEL:-Llama-3.2-3B}"

TRAIN_FILE="${TRAIN_FILE:-['${HOME_DIR}/data/math8k/math8k_hard_solutions_1000.parquet']}"
TEST_FILE="${TEST_FILE:-['${HOME_DIR}/data/math8k/test_solutions_50.parquet']}"
CUSTOM_DATASET="${HOME_DIR}/agentpo/rl_dataset.py"
CKPTS_DIR="${CKPT_ROOT}/${PROJECT_NAME}/${EXP_NAME}"

dataset_num="${DATASET_NUM:-500}"
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-2}"
gen_prompt_bsz="${GEN_PROMPT_BSZ:-2}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-4}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-2}"
max_prompt_length="${MAX_PROMPT_LENGTH:-256}"
max_response_length="${MAX_RESPONSE_LENGTH:-512}"
max_num_gen_batches="${MAX_NUM_GEN_BATCHES:-4}"

mad_num_agents="${MAD_NUM_AGENTS:-2}"
mad_debate_rounds="${MAD_DEBATE_ROUNDS:-1}"
mad_max_tokens="${MAD_MAX_TOKENS:-512}"
mad_max_peer_chars="${MAD_MAX_PEER_CHARS:-600}"
mad_max_concurrency="${MAD_MAX_CONCURRENCY:-4}"
mad_parallel_rollouts="${MAD_PARALLEL_ROLLOUTS:-True}"

n_gpus_per_node="${N_GPUS_PER_NODE:-2}"
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-16}"

python3 -m agentpo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=problem \
    data.truncation=left \
    data.custom_cls.path="${CUSTOM_DATASET}" \
    data.custom_cls.name=RLHFCustomDataset \
    data.dataset_num="${dataset_num}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.return_raw_chat=False \
    data.gen_batch_size="${gen_prompt_bsz}" \
    data.train_batch_size="${train_prompt_bsz}" \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    algorithm.adv_estimator=grpo \
    algorithm.cooperation_mode=mad \
    +algorithm.mad.backend=api \
    +algorithm.mad.vllm_model_path="${MODEL_PATH}" \
    +algorithm.mad.num_agents="${mad_num_agents}" \
    +algorithm.mad.debate_rounds="${mad_debate_rounds}" \
    +algorithm.mad.topology=full \
    +algorithm.mad.multi_persona=False \
    +algorithm.mad.max_tokens="${mad_max_tokens}" \
    +algorithm.mad.max_peer_chars="${mad_max_peer_chars}" \
    +algorithm.mad.parallel_agents=True \
    +algorithm.mad.parallel_rollouts="${mad_parallel_rollouts}" \
    +algorithm.mad.max_concurrency="${mad_max_concurrency}" \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=False \
    algorithm.filter_groups.max_num_gen_batches="${max_num_gen_batches}" \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_focal_weight=False \
    actor_rollout_ref.actor.use_balance_weight=False \
    actor_rollout_ref.actor.use_thres_weight=False \
    actor_rollout_ref.actor.focal_gamma=2.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_memory_utilization}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="${actor_ppo_max_token_len}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.max_num_seqs="${rollout_max_num_seqs}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    custom_reward_function.path="${HOME_DIR}/agentpo/reward_fn.py" \
    reward_model.reward_manager=agentpo \
    reward_model.actor_model="${ACTOR_MODEL}" \
    reward_model.overlong_buffer.enable=False \
    reward_model.overlong_buffer.len=4096 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=100 \
    trainer.save_freq=100 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto

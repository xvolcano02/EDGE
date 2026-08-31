#!/usr/bin/env bash

if [[ "${DEBUG_SHELL:-false}" == "true" ]]; then
    set -x
fi
set -euo pipefail

ENGINE=${1:-vllm}
if [[ $# -gt 0 ]]; then
    shift  # Remove the engine so "$@" only contains Hydra overrides.
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENABLE_EXPERIENCE_EVOLUTION=${ENABLE_EXPERIENCE_EVOLUTION:-true}
ENABLE_EXP_ROLLOUT=${ENABLE_EXP_ROLLOUT:-true}
ENABLE_DYNAMIC_MANAGEMENT=${ENABLE_DYNAMIC_MANAGEMENT:-true}
EXP_ROLLOUT_WARMUP_STEPS=${EXP_ROLLOUT_WARMUP_STEPS:-100}
RETRIEVAL_MODE=${RETRIEVAL_MODE:-embedding}
EXPERIENCE_RETRIEVAL_SERVICE_URL=${EXPERIENCE_RETRIEVAL_SERVICE_URL:-}
EDGE_TOKEN_SELECTION=${EDGE_TOKEN_SELECTION:-sampled}
EDGE_TOP_K=${EDGE_TOP_K:-20}

# Credentials are inherited from the caller's environment and are never passed
# on the command line, where they could be exposed through process listings.
EXPERIENCE_UPDATE_API_KEY=${EXPERIENCE_UPDATE_API_KEY:-}
EXPERIENCE_UPDATE_BASE_URL=${EXPERIENCE_UPDATE_BASE_URL:-}
EXPERIENCE_UPDATE_MODEL=${EXPERIENCE_UPDATE_MODEL:-}
if [[ "$ENABLE_EXPERIENCE_EVOLUTION" == "true" && ( -z "$EXPERIENCE_UPDATE_API_KEY" || -z "$EXPERIENCE_UPDATE_BASE_URL" || -z "$EXPERIENCE_UPDATE_MODEL" ) ]]; then
    echo "Set EXPERIENCE_UPDATE_API_KEY, EXPERIENCE_UPDATE_BASE_URL, and EXPERIENCE_UPDATE_MODEL to enable experience evolution." >&2
    exit 1
fi
if [[ -n "$EXPERIENCE_RETRIEVAL_SERVICE_URL" && "$RETRIEVAL_MODE" != "embedding" ]]; then
    echo "EXPERIENCE_RETRIEVAL_SERVICE_URL requires RETRIEVAL_MODE=embedding." >&2
    exit 1
fi
if [[ "$EDGE_TOKEN_SELECTION" != "sampled" && "$EDGE_TOKEN_SELECTION" != "teacher_topk" ]]; then
    echo "EDGE_TOKEN_SELECTION must be sampled or teacher_topk; got $EDGE_TOKEN_SELECTION." >&2
    exit 1
fi
if [[ "$EDGE_TOKEN_SELECTION" == "teacher_topk" ]] && ! [[ "$EDGE_TOP_K" =~ ^[0-9]+$ && "$EDGE_TOP_K" -ge 2 ]]; then
    echo "EDGE_TOP_K must be an integer >= 2 when EDGE_TOKEN_SELECTION=teacher_topk; got $EDGE_TOP_K." >&2
    exit 1
fi

REMOTE_RETRIEVAL_ARGS=()
if [[ -n "$EXPERIENCE_RETRIEVAL_SERVICE_URL" ]]; then
    REMOTE_RETRIEVAL_ARGS+=("env.experience_memory.experience_retrieval_service_url=$EXPERIENCE_RETRIEVAL_SERVICE_URL")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_NAME="${WANDB_NAME:-search_edge_qwen2.5_7b}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local model or Hugging Face model identifier.}"
TRAIN_DATA="${TRAIN_DATA:?Set TRAIN_DATA to the training parquet file.}"
VAL_DATA="${VAL_DATA:?Set VAL_DATA to the validation parquet file.}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EXPERIENCES_JSON="${EXPERIENCES_JSON:-$REPO_ROOT/memory_data/search/state_aware_experiences_v2.json}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8030/retrieve}"

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.1}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-256}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-512}
GROUP_SIZE=${GROUP_SIZE:-4}

if [[ "$ENABLE_EXP_ROLLOUT" == "true" ]] && (( GROUP_SIZE % 2 != 0 )); then
    echo "GROUP_SIZE must be even when ENABLE_EXP_ROLLOUT=true; got $GROUP_SIZE." >&2
    exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size="$TRAIN_DATA_SIZE" \
    data.val_batch_size="$VAL_DATA_SIZE" \
    data.max_prompt_length=5000 \
    data.max_response_length=700 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_MODEL_PARALLEL_SIZE" \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    actor_rollout_ref.actor.edge_distillation.enable=True \
    actor_rollout_ref.actor.edge_distillation.coef=1.0 \
    actor_rollout_ref.actor.edge_distillation.token_selection="$EDGE_TOKEN_SELECTION" \
    actor_rollout_ref.actor.edge_distillation.top_k="$EDGE_TOP_K" \
    actor_rollout_ref.actor.edge_distillation.require_positive_gain=True \
    actor_rollout_ref.actor.edge_distillation.target=w_o_experience_only \
    algorithm.use_kl_in_reward=False \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n="$GROUP_SIZE" \
    env.history_length=4 \
    env.search.search_url="$SEARCH_URL" \
    env.resources_per_worker.num_cpus="$NUM_CPUS_PER_ENV_WORKER" \
    env.use_experience_memory=True \
    env.experience_memory.experiences_json_path="$EXPERIENCES_JSON" \
    env.experience_memory.retrieval_mode="$RETRIEVAL_MODE" \
    env.experience_memory.embedding_model_path="$EMBEDDING_MODEL" \
    "${REMOTE_RETRIEVAL_ARGS[@]}" \
    env.experience_memory.top_k=6 \
    env.experience_memory.task_specific_top_k=4 \
    env.experience_memory.enable_exp_rollout="$ENABLE_EXP_ROLLOUT" \
    env.experience_memory.exp_rollout_warmup_steps="$EXP_ROLLOUT_WARMUP_STEPS" \
    env.experience_memory.inject_memory_in_val=False \
    env.experience_memory.enable_experience_evolution="$ENABLE_EXPERIENCE_EVOLUTION" \
    env.experience_memory.enable_dynamic_management="$ENABLE_DYNAMIC_MANAGEMENT" \
    env.experience_memory.update_threshold=0.3 \
    env.experience_memory.update_frequency=5 \
    env.experience_memory.max_new_experiences=3 \
    env.experience_memory.max_failed_trajectories=10 \
    env.experience_memory.experience_update_base_url="$EXPERIENCE_UPDATE_BASE_URL" \
    env.experience_memory.experience_update_model="$EXPERIENCE_UPDATE_MODEL" \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.log_val_generations=10 \
    trainer.project_name=verl_agent_search \
    trainer.experiment_name=edge_qwen3_4b_experience_distillation \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.save_best_checkpoint=True \
    trainer.best_checkpoint_metric=val/success_rate \
    trainer.test_freq=200 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    "$@"

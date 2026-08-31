#!/usr/bin/env bash
# Launch the remote experience retrieval service.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "$CONDA_ENV" && ! -f "$CONDA_ENV/bin/activate" ]]; then
  echo "CONDA_ENV does not contain bin/activate: $CONDA_ENV" >&2
  exit 1
fi
if [[ -n "$CONDA_ENV" ]]; then
  source "$CONDA_ENV/bin/activate"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra _DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${NUM_GPUS:-${#_DEVICES[@]}}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EXPERIENCES_JSON="${EXPERIENCES_JSON:-$REPO_ROOT/memory_data/alfworld/state_aware_experiences_v2.json}"
EXPERIENCE_TEXT_FOR_RETRIEVAL="${EXPERIENCE_TEXT_FOR_RETRIEVAL:-full}"

ARGS=(
  --host "$HOST"
  --port "$PORT"
  --device cuda
  --num_gpus "$NUM_GPUS"
  --embedding_model_path "$EMBEDDING_MODEL"
  --experience_text_for_retrieval "$EXPERIENCE_TEXT_FOR_RETRIEVAL"
)

if [[ "${NO_LOAD_INITIAL_EXPERIENCES:-0}" == "1" ]]; then
  ARGS+=(--no_load_initial_experiences)
else
  ARGS+=(--experiences_json_path "$EXPERIENCES_JSON")
fi

python examples/grpo_trainer/experience_retrieval_server.py "${ARGS[@]}" "$@"

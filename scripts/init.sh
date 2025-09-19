#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --linear-method   spec_norm|sv_clip|normal_0p02|normal_6e-3  (default: spec_norm)
  --embed-method    default|std_0p25                          (default: default)
  --head-method     default|30_over_d                         (default: default)
  --per-head        1|0                                       (default: 0)
  --wandb           1|0                                       (default: 1)
  --project         <project_name>                            (default: speedrun_finelog)
  --mode            online|offline|disabled                   (default: online)
  --nproc           <num_gpus>                                (default: 8)
  --tag             <experiment_tag>                          (default: none)
  --dryrun          Print command only
  -h|--help         Show this help

Example:
  $0 --linear-method spec_norm --embed-method std_0p25 --tag test
EOF
}

# Defaults
LINEAR_METHOD="spec_norm"
EMBED_METHOD="default"
HEAD_METHOD="default"
PER_HEAD="0"
WANDB="1"
WANDB_PROJECT="speedrun_finelog"
WANDB_MODE="online"
NPROC="8"
TAG=""
DRYRUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --linear-method)  LINEAR_METHOD="$2"; shift 2;;
    --embed-method)   EMBED_METHOD="$2"; shift 2;;
    --head-method)    HEAD_METHOD="$2"; shift 2;;
    --per-head)       PER_HEAD="$2"; shift 2;;
    --wandb)          WANDB="$2"; shift 2;;
    --project)        WANDB_PROJECT="$2"; shift 2;;
    --mode)           WANDB_MODE="$2"; shift 2;;
    --nproc)          NPROC="$2"; shift 2;;
    --tag)            TAG="$2"; shift 2;;
    --dryrun)         DRYRUN="1"; shift 1;;
    -h|--help)        usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

# Build experiment name
NAME="${LINEAR_METHOD}_${EMBED_METHOD}_${HEAD_METHOD}"
[[ "$PER_HEAD" == "1" ]] && NAME="${NAME}_perhead"
[[ -n "$TAG" ]] && NAME="${NAME}_${TAG}"

# Set environment variables
export INIT_LINEAR_METHOD="$LINEAR_METHOD"
export INIT_PER_HEAD="$PER_HEAD"
export INIT_EMBED_METHOD="$EMBED_METHOD"
export INIT_HEAD_METHOD="$HEAD_METHOD"
export SPEC_LOG_EVERY="25"
export LOG_IO_RMS="1"
export TORCH_COMPILE="1"
export ENABLE_WANDB="$WANDB"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_MODE="$WANDB_MODE"
export EXP_NAME="$NAME"

[[ "$WANDB_MODE" == "disabled" ]] && export ENABLE_WANDB="0"

CMD="torchrun --standalone --nproc_per_node=$NPROC train_gpt.py"

echo "Starting: $NAME"
if [[ "$DRYRUN" == "1" ]]; then
  echo "$CMD"
else
  exec $CMD
fi


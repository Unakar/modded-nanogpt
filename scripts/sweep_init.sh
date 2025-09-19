#!/usr/bin/env bash
set -euo pipefail

# Sweep logic: 5 linear methods × 2 embed/head pairs × N repeats = total runs

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --project        <wandb_project>           (default: sweep_speedrun_init)
  --entity         <wandb_entity>
  --mode           online|offline|disabled   (default: online)
  --repeats        <int>                     (default: 10)
  --nproc          <int>                     (default: 8)
  -h|--help        Show this help

Example: $0 --project my_sweep --repeats 5
EOF
}

WANDB_PROJECT="sweep_speedrun_init"
WANDB_ENTITY=""
WANDB_MODE="online"
COMPILE="1"
LOG_IO="1"
SPEC_LOG_EVERY="25"
REPEATS=10
NPROC="8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)        WANDB_PROJECT="$2"; shift 2;;
    --entity)         WANDB_ENTITY="$2"; shift 2;;
    --mode)           WANDB_MODE="$2"; shift 2;;
    --compile)        COMPILE="$2"; shift 2;;
    --log-io)         LOG_IO="$2"; shift 2;;
    --spec-log-every) SPEC_LOG_EVERY="$2"; shift 2;;
    --repeats)        REPEATS="$2"; shift 2;;
    --nproc)          NPROC="$2"; shift 2;;
    -h|--help)        usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

linears=(
  default
  spec_norm
  sv_clip
  normal_0p02
  normal_6e-3
)

eh_names=(default scaled)
eh_embed=(default std_0p25)
eh_head=( default 30_over_d)

for LM in "${linears[@]}"; do
  for i in "${!eh_names[@]}"; do
    EM=${eh_embed[$i]}
    HM=${eh_head[$i]}
    EHNM=${eh_names[$i]}
    # Set shared group for aggregation of repeats in WandB
    export WANDB_GROUP="linear=${LM};eh=${EHNM}"
    for (( r=0; r<REPEATS; r++ )); do
      TAG="${EHNM}_rep${r}"
      echo "==> linear=$LM | EH=$EHNM (embed=$EM head=$HM) | repeat $r/$((REPEATS-1))"
      args=(
        --linear-method "$LM"
        --embed-method "$EM"
        --head-method "$HM"
        --per-head 0
        --project "$WANDB_PROJECT"
        --mode "$WANDB_MODE"
        --tag "$TAG"
        --nproc "$NPROC"
        --spec-log-every "$SPEC_LOG_EVERY"
        --log-io "$LOG_IO"
        --compile "$COMPILE"
      )
      if [[ -n "$WANDB_ENTITY" ]]; then
        args+=( --entity "$WANDB_ENTITY" )
      fi
      bash scripts/init.sh "${args[@]}"
    done
  done
done


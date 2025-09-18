#!/usr/bin/env bash
set -euo pipefail

# Simple, clear launcher for init experiments on train_gpt.py.
# - Supports single runs and quick sweeps (comma-separated values)
# - Builds informative EXP_NAMEs for easy comparison
# - Keeps options focused on init/logging; not overly verbose

usage() {
  cat <<EOF
Usage: $0 [options]

Init options (comma-separated values allowed for sweeps):
  --linear-method   default|spec_norm|sv_clip|normal_0p02|normal_6e-3  (default: default)
  --embed-method    default|std_0p25                                   (default: default)
  --head-method     default|30_over_d                                   (default: default)
  --per-head        1|0                                                 (default: 1)

Logging:
  --wandb           1|0                                                 (default: 1)
  --project         <wandb_project>                                     (default: modded-nanogpt)
  --mode            online|offline|disabled                             (default: offline)
  --spec-log-every  <int>                                               (default: 25)
  --log-io          1|0                                                 (default: 1)

Engine:
  --compile         1|0                                                 (default: 1)
  --nproc           <int GPUs>                                          (default: 8)

Misc:
  --tag             Freeform tag appended to EXP_NAME                   (default: none)
  --dryrun          Print commands only (no execution)

Examples:
  # Single run (per-head spec_norm, embed std 0.25, head 30/d)
  $0 --linear-method spec_norm --embed-method std_0p25 --head-method 30_over_d --per-head 1 --tag bmup

  # Sweep linear init methods (4x runs) with same embed/head
  $0 --linear-method spec_norm,sv_clip,normal_0p02,normal_6e-3 \
     --embed-method std_0p25 --head-method 30_over_d --per-head 1 --tag sweep
EOF
}

# Defaults
LINEAR_METHODS="spec_norm"
EMBED_METHODS="default"
HEAD_METHODS="default"
PER_HEAD="0"
WANDB="1"
WANDB_PROJECT="speedrun_finelog"
WANDB_MODE="online"   # online|offline|disabled
SPEC_LOG_EVERY="25"
LOG_IO="1"
COMPILE="1"
NPROC="8"
TAG=""
DRYRUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --linear-method)    LINEAR_METHODS="$2"; shift 2;;
    --embed-method)     EMBED_METHODS="$2"; shift 2;;
    --head-method)      HEAD_METHODS="$2"; shift 2;;
    --per-head)         PER_HEAD="$2"; shift 2;;
    --wandb)            WANDB="$2"; shift 2;;
    --project)          WANDB_PROJECT="$2"; shift 2;;
    --mode)             WANDB_MODE="$2"; shift 2;;
    --spec-log-every)   SPEC_LOG_EVERY="$2"; shift 2;;
    --log-io)           LOG_IO="$2"; shift 2;;
    --compile)          COMPILE="$2"; shift 2;;
    --nproc)            NPROC="$2"; shift 2;;
    --tag)              TAG="$2"; shift 2;;
    --dryrun)           DRYRUN="1"; shift 1;;
    -h|--help)          usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

IFS=',' read -r -a LM_ARR <<< "$LINEAR_METHODS"
IFS=',' read -r -a EM_ARR <<< "$EMBED_METHODS"
IFS=',' read -r -a HM_ARR <<< "$HEAD_METHODS"

ts() { date +"%m%d-%H%M%S"; }

for LM in "${LM_ARR[@]}"; do
  for EM in "${EM_ARR[@]}"; do
    for HM in "${HM_ARR[@]}"; do
      # Create readable experiment name with meaningful abbreviations
      LM_SHORT=""
      case "$LM" in
        "default") LM_SHORT="def";;
        "spec_norm") LM_SHORT="spec";;
        "sv_clip") LM_SHORT="clip";;
        "normal_0p02") LM_SHORT="n002";;
        "normal_6e-3") LM_SHORT="n006";;
        *) LM_SHORT="$LM";;
      esac

      EM_SHORT=""
      case "$EM" in
        "default") EM_SHORT="def";;
        "std_0p25") EM_SHORT="s25";;
        *) EM_SHORT="$EM";;
      esac

      HM_SHORT=""
      case "$HM" in
        "default") HM_SHORT="def";;
        "30_over_d") HM_SHORT="30d";;
        *) HM_SHORT="$HM";;
      esac

      PH_SHORT=""
      case "$PER_HEAD" in
        "1") PH_SHORT="ph";;
        "0") PH_SHORT="noph";;
        *) PH_SHORT="ph$PER_HEAD";;
      esac

      # Build clean experiment name: init_method_combination_timestamp_tag
      NAME="linear-${LM_SHORT}_emb-${EM_SHORT}_lmhead${HM_SHORT}_qkvsplit-${PH_SHORT}"
      if [[ -n "$TAG" && "$TAG" != "default-test" ]]; then
        NAME="${NAME}_${TAG}"
      fi

      # Map WANDB_MODE disabled -> ENABLE_WANDB=0
      ENABLE_WANDB="$WANDB"
      MODE="$WANDB_MODE"
      if [[ "$MODE" == "disabled" ]]; then ENABLE_WANDB="0"; MODE="offline"; fi

      export INIT_LINEAR_METHOD="$LM"
      export INIT_PER_HEAD="$PER_HEAD"
      export INIT_EMBED_METHOD="$EM"
      export INIT_HEAD_METHOD="$HM"
      export SPEC_LOG_EVERY="$SPEC_LOG_EVERY"
      export LOG_IO_RMS="$LOG_IO"
      export TORCH_COMPILE="$COMPILE"
      export ENABLE_WANDB="$ENABLE_WANDB"
      export WANDB_PROJECT="$WANDB_PROJECT"
      export WANDB_MODE="$MODE"
      export EXP_NAME="$NAME"

      CMD=(torchrun --standalone --nproc_per_node="$NPROC" train_gpt.py)

      echo "==============================================="
      echo "🚀 Starting Experiment: $NAME"
      echo "==============================================="
      echo "📋 Initialization Methods:"
      echo "   Linear:     $LM"
      echo "   Embedding:  $EM"
      echo "   Head:       $HM"
      echo "   Per-head:   $PER_HEAD"
      echo ""
      echo "🔧 Training Config:"
      echo "   GPUs:       $NPROC"
      echo "   Compile:    $COMPILE"
      echo "   IO RMS:     $LOG_IO"
      echo "   Spec Log:   every $SPEC_LOG_EVERY steps"
      echo ""
      echo "📊 Logging:"
      echo "   WandB:      $ENABLE_WANDB ($MODE)"
      echo "   Project:    $WANDB_PROJECT"
      echo "==============================================="
      if [[ "$DRYRUN" == "1" ]]; then
        continue
      fi
      "${CMD[@]}"
    done
  done
done


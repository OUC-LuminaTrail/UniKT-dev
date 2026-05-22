#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
DATA="$ROOT/data"
LOG_ROOT="${LOG_ROOT:-$ROOT/exp_logs}"
DATASETS="${DATASETS:-assistments12 assistments15 assistments17 slepemapy ednet_kt1 junyi2015 assistments09}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
DOWNLOAD_THREADS="${DOWNLOAD_THREADS:-4}"
EDNET_RAW_LIMIT="${EDNET_RAW_LIMIT:-12000}"

ASSIST09_RAW="${ASSIST09_RAW:-/root/autodl-tmp/kt-exp-graph/data/assistments09/raw/skill_builder_data_corrected_collapsed.csv}"
ASSIST12_RAW="${ASSIST12_RAW:-/root/autodl-tmp/kt-exp-graph/data/assistments12/raw/2012-2013-data-with-predictions-4-final.csv}"
ASSIST15_RAW="${ASSIST15_RAW:-/root/autodl-tmp/pykt-toolkit/data/assist2015/2015_100_skill_builders_main_problems.csv}"
EDNET_CONTENTS_RAW="${EDNET_CONTENTS_RAW:-/root/autodl-tmp/pykt-toolkit/data/ednet/contents/questions.csv}"
EDNET_KT1_RAW_DIR="${EDNET_KT1_RAW_DIR:-/root/autodl-tmp/pykt-toolkit/data/ednet/KT1}"

cd "$ROOT"
mkdir -p "$DATA" "$LOG_ROOT"
LOGDIR="$LOG_ROOT/robustkt_fresh_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/summary.log"
}

require_file() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    echo "Required file missing: $path" >&2
    return 1
  fi
}

require_dir() {
  local path=$1
  if [[ ! -d "$path" ]]; then
    echo "Required directory missing: $path" >&2
    return 1
  fi
}

clean_archives_and_raw() {
  local ds=$1
  rm -rf "$DATA/$ds/raw"
  find "$DATA/$ds" -maxdepth 1 -type f \
    \( -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" -o -name "*.gz" \) \
    -delete 2>/dev/null || true
}

clean_processed_files() {
  local ds=$1
  mkdir -p "$DATA/$ds"
  rm -f "$DATA/$ds"/*_sequence.parquet \
        "$DATA/$ds"/*_split_question_sequence.parquet \
        "$DATA/$ds"/*_split_skill_sequence.parquet \
        "$DATA/$ds"/*_windowlate.parquet \
        "$DATA/$ds"/*_relation_*.parquet \
        "$DATA/$ds"/*_question.parquet \
        "$DATA/$ds"/metadata.json
}

copy_or_download_raw() {
  local ds=$1
  clean_processed_files "$ds"
  mkdir -p "$DATA/$ds/raw"

  case "$ds" in
    assistments09)
      require_file "$ASSIST09_RAW"
      cp "$ASSIST09_RAW" "$DATA/$ds/raw/"
      ;;
    assistments12)
      require_file "$ASSIST12_RAW"
      cp "$ASSIST12_RAW" "$DATA/$ds/raw/"
      ;;
    assistments15)
      require_file "$ASSIST15_RAW"
      cp "$ASSIST15_RAW" "$DATA/$ds/raw/"
      ;;
    ednet_kt1)
      require_file "$EDNET_CONTENTS_RAW"
      require_dir "$EDNET_KT1_RAW_DIR"
      rm -rf "$DATA/$ds/raw"
      mkdir -p "$DATA/$ds/raw/EdNet-Contents" "$DATA/$ds/raw/EdNet-KT1/KT1"
      cp "$EDNET_CONTENTS_RAW" "$DATA/$ds/raw/EdNet-Contents/questions.csv"
      find "$EDNET_KT1_RAW_DIR" -maxdepth 1 -type f -name "u*.csv" \
        | sort | sed -n "1,${EDNET_RAW_LIMIT}p" \
        | while IFS= read -r file; do
            cp "$file" "$DATA/$ds/raw/EdNet-KT1/KT1/"
          done
      log "EDNET_RAW_FILES=$(find "$DATA/$ds/raw/EdNet-KT1/KT1" -maxdepth 1 -type f -name 'u*.csv' | wc -l)"
      ;;
    assistments17|slepemapy|junyi2015)
      "$PY" data_process.py download \
        -d "$ds" \
        --force \
        --num_threads "$DOWNLOAD_THREADS" \
        --data_base_path "$DATA" \
        2>&1 | tee "$LOGDIR/${ds}_download.log"
      ;;
    *)
      echo "Unknown dataset: $ds" >&2
      return 1
      ;;
  esac
}

process_dataset() {
  local ds=$1
  local sample_args=()
  if [[ "$ds" == "slepemapy" || "$ds" == "ednet_kt1" || "$ds" == "junyi2015" ]]; then
    sample_args=(--sample_size 5000 --sample_strategy random)
  fi

  log "PROCESS_START dataset=$ds sample_args=${sample_args[*]-}"
  log "PROCESS_COMMAND $PY data_process.py process -d $ds --extra windowlate --data_base_path $DATA ${sample_args[*]-}"
  copy_or_download_raw "$ds"
  du -h -d 1 "$DATA/$ds" | tee -a "$LOGDIR/summary.log"

  "$PY" data_process.py process \
    -d "$ds" \
    --extra windowlate \
    --data_base_path "$DATA" \
    "${sample_args[@]}" \
    2>&1 | tee "$LOGDIR/${ds}_process.log"

  "$PY" - <<PY | tee -a "$LOGDIR/summary.log"
import json
from pathlib import Path

metadata_path = Path("$DATA/$ds/metadata.json")
metadata = json.loads(metadata_path.read_text())
keys = [
    "dataset",
    "data_base_path",
    "num_users",
    "num_split_skill_users",
    "num_questions",
    "num_skills",
    "kfold_n_splits",
    "test_ratio",
    "sampled",
    "sample_size",
    "sample_strategy",
    "sampling_config",
    "sampling_stats",
    "windowlate_data_md5",
]
print("METADATA", "$ds", {key: metadata.get(key) for key in keys})
PY

  log "CLEAN_RAW_AND_ARCHIVES dataset=$ds"
  clean_archives_and_raw "$ds"
  du -h -d 1 "$DATA/$ds" | tee -a "$LOGDIR/summary.log"
  df -h / /root/autodl-tmp 2>/dev/null | tee -a "$LOGDIR/summary.log" || df -h "$DATA" | tee -a "$LOGDIR/summary.log"
}

train_fold() {
  local ds=$1
  local fold=$2
  log "TRAIN_START dataset=$ds fold=$fold"
  log "TRAIN_COMMAND $PY train.py -m RobustKT -d $ds --fold $fold --epochs $EPOCHS --batch_size $BATCH_SIZE --test_batch_size $TEST_BATCH_SIZE --deterministic --data_base_path $DATA"
  "$PY" train.py \
    -m RobustKT \
    -d "$ds" \
    --fold "$fold" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --deterministic \
    --data_base_path "$DATA" \
    2>&1 | tee "$LOGDIR/${ds}_fold${fold}.log"
  log "TRAIN_DONE dataset=$ds fold=$fold"
}

run_dataset() {
  local ds=$1
  process_dataset "$ds"
  for fold in $FOLDS; do
    train_fold "$ds" "$fold"
  done
  log "DATASET_DONE dataset=$ds"
  df -h / /root/autodl-tmp 2>/dev/null | tee -a "$LOGDIR/summary.log" || df -h "$DATA" | tee -a "$LOGDIR/summary.log"
}

log "ROOT=$ROOT"
log "DATA=$DATA"
log "LOGDIR=$LOGDIR"
log "COMMIT=$(git rev-parse HEAD)"
log "DATASETS=$DATASETS"
log "FOLDS=$FOLDS"
log "EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE TEST_BATCH_SIZE=$TEST_BATCH_SIZE"
df -h / /root/autodl-tmp 2>/dev/null | tee -a "$LOGDIR/summary.log" || df -h "$DATA" | tee -a "$LOGDIR/summary.log"

for ds in $DATASETS; do
  run_dataset "$ds"
done

log "ALL_DONE"

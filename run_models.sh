#!/usr/bin/env bash
# Runs all evaluation commands for each model in a provided list.
# Usage: ./run_all.sh models.txt [num_parallel_request] [stated_prefs_script] [with_definitions]
# Env:   OPENROUTER_API_KEY must be set.

set -o pipefail

if [[ -z "$1" ]]; then
  echo "Usage: $0 <models_file> [num_parallel_request] [stated_prefs_script] [with_definitions]"
  exit 1
fi

MODELS_FILE="$1"
NUM_PARALLEL_REQUEST="${2:-80}"
STATED_PREFS_SCRIPT="${3:-run_stated_preferences.py}"
WITH_DEFINITIONS="$4"

# Set output directories based on the script used and whether definitions are included
if [[ "$STATED_PREFS_SCRIPT" == "run_stated_preferences_controlled.py" ]]; then
  if [[ "$WITH_DEFINITIONS" == "with_definitions" ]]; then
    GENERATIONS_STATED_DIR="generations_stated_controlled_with_defs"
    ELO_RATING_STATED_DIR="elo_rating_stated_controlled_with_defs"
  else
    GENERATIONS_STATED_DIR="generations_stated_controlled"
    ELO_RATING_STATED_DIR="elo_rating_stated_controlled"
  fi
else
  if [[ "$WITH_DEFINITIONS" == "with_definitions" ]]; then
    GENERATIONS_STATED_DIR="generations_stated_with_defs"
    ELO_RATING_STATED_DIR="elo_rating_stated_with_defs"
  else
    GENERATIONS_STATED_DIR="generations_stated"
    ELO_RATING_STATED_DIR="elo_rating_stated"
  fi
fi

GENERATIONS_REVEALED_DIR="generations"
ELO_RATING_REVEALED_DIR="elo_rating"

if [[ ! -f "$MODELS_FILE" ]]; then
  echo "Models file not found: $MODELS_FILE"
  exit 1
fi

if [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set in your environment."
  echo "Export it first, e.g.: export OPENROUTER_API_KEY='sk-or-...'"
  exit 1
fi

mkdir -p logs "$GENERATIONS_REVEALED_DIR" "$ELO_RATING_REVEALED_DIR"

sanitize() {
  # Turn model IDs into safe file-name tokens
  # e.g. meta-llama/llama-3.3-70b-instruct -> meta-llama_llama-3.3-70b-instruct
  echo "$1" | tr '/:' '__'
}

run_for_model() {
  local model="$1"
  local safe
  safe="$(sanitize "$model")"

  # Create model-specific directories
  mkdir -p "$GENERATIONS_REVEALED_DIR" "$ELO_RATING_REVEALED_DIR" "$ELO_RATING_STATED_DIR"

  echo "────────────────────────────────────────"
  echo "▶ Model: $model"
  echo "────────────────────────────────────────"

  # # 1) AI Risk Dilemmas (Revealed Preferences)
  # echo "[${model}] Running run_ai_risk_dilemmas.py..."
  # python run_ai_risk_dilemmas.py \
  #   --api_provider openrouter \
  #   --model "$model" \
  #   --api_key "$OPENROUTER_API_KEY" \
  #   --num_parallel_request "$NUM_PARALLEL_REQUEST" \
  #   --generations_dir "$GENERATIONS_REVEALED_DIR" \
  #   2>&1 | tee "logs/${safe}_ai_risk_dilemmas.log"
  # status1=${PIPESTATUS[0]}

  # 2) Stated Preferences
  echo "[${model}] Running $STATED_PREFS_SCRIPT..."
  if [[ "$WITH_DEFINITIONS" == "with_definitions" ]]; then
    python "$STATED_PREFS_SCRIPT" \
      --api_provider openrouter \
      --model "$model" \
      --api_key "$OPENROUTER_API_KEY" \
      --num_parallel_request "$NUM_PARALLEL_REQUEST" \
      --output_dir "$GENERATIONS_STATED_DIR" \
      --with_definitions \
      2>&1 | tee "logs/${safe}_stated_prefs_with_defs.log"
  else
    python "$STATED_PREFS_SCRIPT" \
      --api_provider openrouter \
      --model "$model" \
      --api_key "$OPENROUTER_API_KEY" \
      --num_parallel_request "$NUM_PARALLEL_REQUEST" \
      --output_dir "$GENERATIONS_STATED_DIR" \
      2>&1 | tee "logs/${safe}_stated_prefs.log"
  fi
  status2=${PIPESTATUS[0]}

  # 3) Calculate Elo (Revealed Preferences)
  echo "[${model}] Running calculate_elo_rating.py..."
  python calculate_elo_rating.py -m "$model" \
    --generations_dir "$GENERATIONS_REVEALED_DIR" \
    --elo_rating_dir "$ELO_RATING_REVEALED_DIR" \
    2>&1 | tee "logs/${safe}_elo_pairwise.log"
  status3=${PIPESTATUS[0]}

  # 4) Calculate Elo (Stated Preferences)
  echo "[${model}] Running calculate_elo_rating_stated.py..."
  python calculate_elo_rating_stated.py -m "$model" \
    --generations_dir "$GENERATIONS_STATED_DIR" \
    --elo_rating_dir "$ELO_RATING_STATED_DIR" \
    2>&1 | tee "logs/${safe}_elo_stated.log"
  status4=${PIPESTATUS[0]}

  # 5) Visualize Elo (Revealed Preferences)
  echo "[${model}] Running visualize_elo_rating.py..."
  python visualize_elo_rating.py -m "$model" \
    --generations_dir "$GENERATIONS_REVEALED_DIR" \
    --output_elo_fig_dir "output_elo_figs" \
    --output_win_rate_fig_dir "output_win_rate_figs" \
    2>&1 | tee "logs/${safe}_viz_pairwise.log"
  status5=${PIPESTATUS[0]}

  # 6) Visualize Elo (Stated Preferences)
  echo "[${model}] Running visualize_elo_rating_stated.py..."
  python visualize_elo_rating_stated.py -m "$model" \
    --elo_rating_dir "$ELO_RATING_STATED_DIR" \
    --output_elo_fig_dir "output_elo_figs_stated" \
    --output_win_rate_fig_dir "output_win_rate_figs_stated" \
    2>&1 | tee "logs/${safe}_viz_stated.log"
  status6=${PIPESTATUS[0]}

  # Summary
  echo "[${model}] Exit codes: aiRisk=$status1 statedPref=$status2 eloRevealed=$status3 eloStated=$status4 vizRevealed=$status5 vizStated=$status6" \
    | tee -a "logs/${safe}_summary.log"

  # Non-zero exit reporting without stopping the whole run:
  if (( status1 || status2 || status3 || status4 || status5 || status6 )); then
    echo "[${model}] ⚠️ Some steps failed. See logs/${safe}_*.log"
  else
    echo "[${model}] ✅ All steps completed."
  fi

  echo
}

# Allow blank lines and comments starting with '#'
while IFS= read -r line || [[ -n "$line" ]]; do
  # trim
  model="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -z "$model" || "$model" =~ ^# ]] && continue
  run_for_model "$model"
done < "$MODELS_FILE"

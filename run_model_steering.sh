#!/usr/bin/env bash
# Runs steered evaluation and analysis for a given model or list of models.
# Usage: ./run_model_steering.sh <model_name_or_file> [steering_source] [num_parallel_request]
#
# Arguments:
#   1. <model_name_or_file>: Either a single model ID (e.g., "openai/gpt-4o") 
#                            OR a text file containing a list of models.
#   2. [steering_source]: "stated" (default) or "controlled_stated".
#   3. [num_parallel_request]: Number of parallel requests (default: 20).
#
# Env: OPENROUTER_API_KEY must be set.

set -o pipefail

# --- Input Validation ---
if [[ -z "$1" ]]; then
  echo "Usage: $0 <model_name_or_file> [steering_source] [num_parallel_request]"
  echo "Example: $0 models.txt stated 20"
  echo "Example: $0 openai/gpt-4o controlled_stated 10"
  exit 1
fi

INPUT_ARG="$1"
STEERING_SOURCE="${2:-stated}"
NUM_PARALLEL_REQUEST="${3:-20}"

# Determine ELO directories based on steering source
if [[ "$STEERING_SOURCE" == "stated" ]]; then
    ELO_INPUT_DIR="elo_rating_stated"
    STEERED_OUTPUT_DIR="generations_steered_stated" # Matching Python script default
    ELO_RATING_STEERED_DIR="elo_rating_steered_stated"
    OUTPUT_PREFIX="steering_analysis_stated"
elif [[ "$STEERING_SOURCE" == "controlled_stated" ]]; then
    ELO_INPUT_DIR="elo_rating_stated_controlled"
    STEERED_OUTPUT_DIR="generations_steered_controlled_stated"
    ELO_RATING_STEERED_DIR="elo_rating_steered_controlled"
    OUTPUT_PREFIX="steering_analysis_controlled"
else
    echo "Error: steering_source must be 'stated' or 'controlled_stated'"
    exit 1
fi

# Define directory for final plots/analysis
ANALYSIS_OUTPUT_DIR="steering_analysis_results"

# Ensure API Key is set
if [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set."
  echo "Please export it: export OPENROUTER_API_KEY='sk-or-...'"
  exit 1
fi

# Create all necessary directories
mkdir -p logs "$ELO_RATING_STEERED_DIR" "$ANALYSIS_OUTPUT_DIR"

# --- Helper Function to sanitize model names for filenames ---
sanitize() {
  # e.g. meta-llama/llama-3.3-70b-instruct -> meta-llama__llama-3.3-70b-instruct
  echo "$1" | sed 's/\//__/g'
}

# --- Function to run the pipeline for one model ---
run_pipeline() {
    local model="$1"
    local safe_model_name
    safe_model_name="$(sanitize "$model")"

    echo "========================================================"
    echo "▶ Processing Model: $model"
    echo "▶ Steering Source: $STEERING_SOURCE"
    echo "========================================================"

    # 1. Run Steered Generation
    # Outputs to: generations_steered_{source}/{model}.csv
    echo "[${model}] Generating Steered Preferences..."
    
    python run_revealed_preferences_steered.py \
        --api_provider openrouter \
        --api_key "$OPENROUTER_API_KEY" \
        --model "$model" \
        --steering_source "$STEERING_SOURCE" \
        --elo_input_dir "$ELO_INPUT_DIR" \
        --num_parallel_request "$NUM_PARALLEL_REQUEST" \
        2>&1 | tee "logs/${safe_model_name}_steering_generation.log"
    
    status_gen=${PIPESTATUS[0]}

    if [[ $status_gen -ne 0 ]]; then
        echo "❌ Generation failed for $model. Skipping this model."
        return 1
    fi

    # 2. Calculate ELO for the newly generated steered data
    # Outputs to: elo_rating_steered_{source}/{model}.csv
    
    echo "[${model}] Calculating ELO for Steered Preferences..."
    python calculate_elo_rating.py \
        -m "$model" \
        --generations_dir "$STEERED_OUTPUT_DIR" \
        --elo_rating_dir "$ELO_RATING_STEERED_DIR" \
        2>&1 | tee "logs/${safe_model_name}_steering_elo.log"

    status_elo=${PIPESTATUS[0]}

    if [[ $status_elo -ne 0 ]]; then
        echo "❌ ELO calculation failed for $model."
        return 1
    fi

    echo "✅ Processing completed for $model"
    echo
}

# --- Main Logic: Handle File vs Single Model ---

if [[ -f "$INPUT_ARG" ]]; then
    # Input is a file list of models
    echo "Reading models from file: $INPUT_ARG"
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Trim whitespace
        model_line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        # Skip empty lines or comments
        [[ -z "$model_line" || "$model_line" =~ ^# ]] && continue
        
        run_pipeline "$model_line"
    done < "$INPUT_ARG"
else
    # Input is a single model string
    run_pipeline "$INPUT_ARG"
fi

# --- Final Step: Aggregate Analysis ---
# Runs ONCE after all models are processed.

echo "========================================================"
echo "▶ Running Final Aggregate Analysis"
echo "========================================================"

python calculate_steering_improvement.py \
    --stated_dir "$ELO_INPUT_DIR" \
    --revealed_dir "elo_rating" \
    --steered_dir "$ELO_RATING_STEERED_DIR" \
    --output_prefix "${ANALYSIS_OUTPUT_DIR}/${OUTPUT_PREFIX}_final"

echo "✅ Analysis Pipeline Completed."
echo "   Aggregated results saved in: $ANALYSIS_OUTPUT_DIR"
### **How to Use It**

1.  **Save the script** as `run_model_steering.sh`.
2.  **Make it executable:** `chmod +x run_model_steering.sh`.

#### **Scenario A: Run for a single model**
```bash
./run_model_steering.sh "openai/gpt-4o" stated 20
```

#### **Scenario B: Run for a list of models**
Create a `models_steer.txt` file:
```text
openai/gpt-4o
anthropic/claude-3.5-sonnet
meta-llama/llama-3.1-405b-instruct
```
Then run:
```bash
./run_model_steering.sh models_steer.txt stated 20
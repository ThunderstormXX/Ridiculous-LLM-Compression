#!/bin/bash
# scripts/bash/run_window_pruning.sh

MODEL_PATH=$1
WORKSPACE=${2:-"./workspace"}
WINDOW_SIZE=${3:-3}
DEVICES=${4:-"0"}
MAX_STEPS=${5:-1000}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [workspace] [window_size] [devices] [max_steps]"
    echo "  devices: comma-separated GPU IDs (e.g., '0,1,2') or 'cpu'"
    exit 1
fi

# Set CUDA devices
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
if [ "$DEVICES" != "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=$DEVICES
    echo "Using GPU devices: $DEVICES"
else
    export CUDA_VISIBLE_DEVICES=""
    echo "Using CPU"
fi

echo "Starting window pruning for model: $MODEL_PATH"
echo "Workspace: $WORKSPACE"
echo "Window size: $WINDOW_SIZE"
echo "Max fine-tuning steps: $MAX_STEPS"

mkdir -p $WORKSPACE

# Create experiment directory with run number
EXP_DIR="$WORKSPACE/window_pruning"
mkdir -p $EXP_DIR
RUN_NUM=1
while [ -d "$EXP_DIR/run_$RUN_NUM" ]; do
    RUN_NUM=$((RUN_NUM + 1))
done
RUN_DIR="$EXP_DIR/run_$RUN_NUM"
mkdir -p $RUN_DIR

echo "Experiment directory: $RUN_DIR"

# Find unimportant layers
echo "=== Finding least important layer window ==="
python scripts/unimportant_decoder_search.py \
    --model_path $MODEL_PATH \
    --window_size $WINDOW_SIZE \
    --output_file $RUN_DIR/unimportant_layers.json \
    --workspace $RUN_DIR \
    --device auto

# Extract window from results
WINDOW=$(python -c "
import json
with open('$RUN_DIR/unimportant_layers.json', 'r') as f:
    data = json.load(f)
print(','.join(map(str, data['best_window'])))
")

echo "Least important window: layers $WINDOW"

# Prune the window and fine-tune
echo "=== Pruning window layers and fine-tuning ==="
python scripts/window_pruning.py \
    --model_path $MODEL_PATH \
    --layers $WINDOW \
    --workspace $RUN_DIR \
    --max_steps $MAX_STEPS \
    --device auto

echo "Window pruning and fine-tuning completed!"
echo "Results saved to: $RUN_DIR"
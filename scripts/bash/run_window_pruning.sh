#!/bin/bash
# scripts/bash/run_window_pruning.sh

MODEL_PATH=$1
WORKSPACE=${2:-"./workspace"}
WINDOW_SIZE=${3:-3}
DEVICES=${4:-"0"}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [workspace] [window_size] [devices]"
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

mkdir -p $WORKSPACE

# Find unimportant layers
echo "=== Finding least important layer window ==="
python scripts/unimportant_decoder_search.py \
    --model_path $MODEL_PATH \
    --window_size $WINDOW_SIZE \
    --output_file $WORKSPACE/unimportant_layers.json \
    --workspace $WORKSPACE \
    --device auto

# Extract window from results
WINDOW=$(python -c "
import json
with open('$WORKSPACE/unimportant_layers.json', 'r') as f:
    data = json.load(f)
print(','.join(map(str, data['best_window'])))
")

echo "Least important window: layers $WINDOW"

# Prune the window
echo "=== Pruning window layers ==="
python scripts/prune_layers.py \
    --model_path $MODEL_PATH \
    --layers $WINDOW \
    --output_path $WORKSPACE/window_pruned_model \
    --workspace $WORKSPACE \
    --device auto

echo "Window pruning completed!"
echo "Pruned model saved to: $WORKSPACE/window_pruned_model"
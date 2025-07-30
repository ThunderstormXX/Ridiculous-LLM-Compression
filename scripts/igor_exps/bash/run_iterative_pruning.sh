#!/bin/bash
# scripts/bash/run_iterative_pruning.sh

MODEL_PATH=$1
WORKSPACE=${2:-"./workspace"}
NUM_LAYERS=${3:-3}
START_LAYER=${4:-0}
DEVICES=${5:-"0"}
MAX_STEPS=${6:-1000}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [workspace] [num_layers] [start_layer] [devices] [max_steps]"
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

echo "Starting iterative pruning for model: $MODEL_PATH"
echo "Workspace: $WORKSPACE"
echo "Number of layers to prune: $NUM_LAYERS"
echo "Starting from layer: $START_LAYER"
echo "Max steps per iteration: $MAX_STEPS"

mkdir -p $WORKSPACE

# Create experiment directory with run number
EXP_DIR="$WORKSPACE/iterative_pruning"
mkdir -p $EXP_DIR
RUN_NUM=1
while [ -d "$EXP_DIR/run_$RUN_NUM" ]; do
    RUN_NUM=$((RUN_NUM + 1))
done
RUN_DIR="$EXP_DIR/run_$RUN_NUM"
mkdir -p $RUN_DIR

echo "Experiment directory: $RUN_DIR"

# Run iterative pruning script
python scripts/iterative_pruning.py \
    --model_path $MODEL_PATH \
    --workspace $RUN_DIR \
    --num_layers $NUM_LAYERS \
    --start_layer $START_LAYER \
    --max_steps $MAX_STEPS \
    --device auto

echo "Iterative pruning completed!"
echo "Results saved to: $RUN_DIR"
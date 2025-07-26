#!/bin/bash
# scripts/bash/run_iterative_pruning.sh

MODEL_PATH=$1
WORKSPACE=${2:-"./workspace"}
NUM_LAYERS=${3:-5}
START_LAYER=${4:-0}
DEVICES=${5:-"0"}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [workspace] [num_layers] [start_layer] [devices]"
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

mkdir -p $WORKSPACE

for i in $(seq 0 $((NUM_LAYERS-1))); do
    LAYER_IDX=$((START_LAYER + i))
    CURRENT_MODEL="$WORKSPACE/model_step_$i"
    NEXT_MODEL="$WORKSPACE/model_step_$((i+1))"
    
    echo "=== Step $((i+1)): Processing layer $LAYER_IDX ==="
    
    # Use previous model or original model for first iteration
    if [ $i -eq 0 ]; then
        INPUT_MODEL=$MODEL_PATH
    else
        INPUT_MODEL=$CURRENT_MODEL
    fi
    
    # Prune layer
    echo "Pruning layer $LAYER_IDX..."
    python scripts/prune_layers.py \
        --model_path $INPUT_MODEL \
        --layers $LAYER_IDX \
        --output_path $NEXT_MODEL \
        --workspace $WORKSPACE \
        --device auto
    
    # Fine-tune with LoRA
    echo "Fine-tuning with LoRA..."
    python scripts/finetuning.py \
        --model_path $NEXT_MODEL \
        --output_path $NEXT_MODEL \
        --workspace $WORKSPACE \
        --max_steps 500 \
        --device auto
    
    echo "Step $((i+1)) completed. Model saved to: $NEXT_MODEL"
done

echo "Iterative pruning completed!"
echo "Final model: $WORKSPACE/model_step_$NUM_LAYERS"
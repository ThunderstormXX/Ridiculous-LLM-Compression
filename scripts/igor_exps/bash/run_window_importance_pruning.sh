#!/bin/bash

# Default values
MODEL_PATH="src/checkpoints/llama3.1-8b"
WORKSPACE="./workspace"
WINDOW_SIZE=7
DEVICES="3"
MAX_STEPS=10
HIDDEN_PATH=""
LAYER_TYPE="mlp"
MAX_SAMPLES=80
MAX_TOKENS=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;
    --workspace=*)
      WORKSPACE="${1#*=}"
      shift
      ;;
    --window_size=*)
      WINDOW_SIZE="${1#*=}"
      shift
      ;;
    --devices=*)
      DEVICES="${1#*=}"
      shift
      ;;
    --max_steps=*)
      MAX_STEPS="${1#*=}"
      shift
      ;;
    --hidden_path=*)
      HIDDEN_PATH="${1#*=}"
      shift
      ;;
    --layer_type=*)
      LAYER_TYPE="${1#*=}"
      shift
      ;;
    --max_samples=*)
      MAX_SAMPLES="${1#*=}"
      shift
      ;;
    --max_tokens=*)
      MAX_TOKENS="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

echo "Usage: $0 [--model_path=path] [--workspace=dir] [--window_size=N] [--devices=N] [--max_steps=N] [--hidden_path=path] [--layer_type=type] [--max_samples=N] [--max_tokens=N]"

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=$DEVICES

# Build command
CMD="python scripts/igor_exps/unified_pruning.py \
    --model_path $MODEL_PATH \
    --workspace $WORKSPACE \
    --method window \
    --strategy importance \
    --window_size $WINDOW_SIZE \
    --max_steps $MAX_STEPS \
    --layer_type $LAYER_TYPE \
    --max_samples $MAX_SAMPLES \
    --max_tokens $MAX_TOKENS \
    --device auto"

# Add hidden_path if provided
if [ -n "$HIDDEN_PATH" ]; then
    CMD="$CMD --hidden_path $HIDDEN_PATH"
fi

eval $CMD
#!/bin/bash

# Default values
MODEL_PATH="src/checkpoints/llama3.1-8b"
WORKSPACE="./workspace"
NUM_LAYERS=3
START_LAYER=19
DEVICES="3"
MAX_STEPS=10

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
    --num_layers=*)
      NUM_LAYERS="${1#*=}"
      shift
      ;;
    --start_layer=*)
      START_LAYER="${1#*=}"
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
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

echo "Usage: $0 [--model_path=path] [--workspace=dir] [--num_layers=N] [--start_layer=N] [--devices=N] [--max_steps=N]"

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=$DEVICES

python scripts/igor_exps/unified_pruning.py \
    --model_path $MODEL_PATH \
    --workspace $WORKSPACE \
    --method iterative \
    --num_layers $NUM_LAYERS \
    --start_layer $START_LAYER \
    --max_steps $MAX_STEPS \
    --device auto
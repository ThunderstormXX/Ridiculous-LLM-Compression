#!/bin/bash

# Default values
MODEL_PATH="src/checkpoints/llama3.1-8b"
WORKSPACE="./workspace"
WINDOW_SIZE=3
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
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

echo "Usage: $0 [--model_path=path] [--workspace=dir] [--window_size=N] [--devices=N] [--max_steps=N]"

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=$DEVICES

python scripts/igor_exps/unified_pruning.py \
    --model_path $MODEL_PATH \
    --workspace $WORKSPACE \
    --method window \
    --window_size $WINDOW_SIZE \
    --max_steps $MAX_STEPS \
    --device auto
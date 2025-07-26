#!/bin/bash
# scripts/bash/run_finetune_window_pruned.sh

PRUNED_MODEL_PATH=$1
WORKSPACE=${2:-"./workspace"}
MAX_STEPS=${3:-1000}

if [ -z "$PRUNED_MODEL_PATH" ]; then
    echo "Usage: $0 <pruned_model_path> [workspace] [max_steps]"
    exit 1
fi

echo "Fine-tuning window-pruned model: $PRUNED_MODEL_PATH"
echo "Workspace: $WORKSPACE"
echo "Max steps: $MAX_STEPS"

mkdir -p $WORKSPACE

# Fine-tune the window-pruned model
echo "=== Fine-tuning window-pruned model ==="
python scripts/finetuning.py \
    --model_path $PRUNED_MODEL_PATH \
    --output_path $WORKSPACE/window_pruned_finetuned \
    --workspace $WORKSPACE \
    --max_steps $MAX_STEPS \
    --learning_rate 2e-4

echo "Fine-tuning completed!"
echo "Fine-tuned model saved to: $WORKSPACE/window_pruned_finetuned"
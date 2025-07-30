# Scripts Usage Guide

## Python Scripts

### `iterative_pruning.py`
Runs complete iterative pruning experiment with logging and visualization support.

```bash
python scripts/iterative_pruning.py \
    --model_path /path/to/model \
    --workspace ./workspace \
    --num_layers 3 \
    --start_layer 0 \
    --max_steps 50 \
    --device auto
```

**Parameters:**
- `--model_path`: Path to model directory (required)
- `--workspace`: Output directory for logs and results (default: `./workspace`)
- `--num_layers`: Number of layers to prune (default: `3`)
- `--start_layer`: Starting layer index (default: `0`)
- `--max_steps`: Training steps per iteration (default: `50`)
- `--device`: Device to use - `auto`, `cpu`, `cuda:0` (default: `auto`)

### `prune_layers.py`
Remove specific decoder layers from a model.

```bash
python scripts/prune_layers.py \
    --model_path /path/to/model \
    --layers 5,10,15 \
    --workspace ./workspace \
    --device auto
```

**Parameters:**
- `--model_path`: Input model path (required)
- `--layers`: Comma-separated layer indices to remove (required)
- `--output_path`: Output model path (default: `workspace/model_name_pruned`)
- `--workspace`: Workspace directory (default: `./workspace`)
- `--device`: Device to use (default: `auto`)

### `finetuning.py`
Fine-tune a model on WikiText dataset.

```bash
python scripts/finetuning.py \
    --model_path /path/to/model \
    --output_path /path/to/output \
    --workspace ./workspace \
    --max_steps 500 \
    --learning_rate 2e-4 \
    --batch_size 2 \
    --device auto
```

**Parameters:**
- `--model_path`: Input model path (required)
- `--output_path`: Output model path (required)
- `--workspace`: Workspace directory (default: `./workspace`)
- `--max_steps`: Maximum training steps (default: `500`)
- `--learning_rate`: Learning rate (default: `2e-4`)
- `--batch_size`: Batch size (default: `2`)
- `--device`: Device to use (default: `auto`)

### `unimportant_decoder_search.py`
Find least important decoder layer windows.

```bash
python scripts/unimportant_decoder_search.py \
    --model_path /path/to/model \
    --window_size 3 \
    --output_file results.json \
    --workspace ./workspace \
    --device auto
```

**Parameters:**
- `--model_path`: Model path (required)
- `--window_size`: Size of layer window (default: `3`)
- `--output_file`: Output JSON file (default: `unimportant_layers.json`)
- `--workspace`: Workspace directory (default: `./workspace`)
- `--device`: Device to use (default: `auto`)

## Bash Scripts

### `run_iterative_pruning.sh`
Complete iterative pruning pipeline.

```bash
./scripts/bash/run_iterative_pruning.sh \
    /path/to/model \
    ./workspace \
    3 \
    0 \
    "0" \
    50
```




**Parameters (positional):**
1. `model_path`: Path to model (required)
2. `workspace`: Output directory (default: `./workspace`)
3. `num_layers`: Layers to prune (default: `3`)
4. `start_layer`: Starting layer (default: `0`)
5. `devices`: GPU IDs like `"0,1,2"` or `"cpu"` (default: `"0"`)
6. `max_steps`: Training steps per iteration (default: `50`)

### `run_window_pruning.sh`
Window-based pruning approach.

```bash
./scripts/bash/run_window_pruning.sh \
    /path/to/model \
    ./workspace \
    3 \
    "0"
```

**Parameters (positional):**
1. `model_path`: Path to model (required)
2. `workspace`: Output directory (default: `./workspace`)
3. `window_size`: Window size (default: `3`)
4. `devices`: GPU IDs (default: `"0"`)

### `run_finetune_window_pruned.sh`
Fine-tune window-pruned models.

```bash
./scripts/bash/run_finetune_window_pruned.sh \
    /path/to/pruned/model \
    ./workspace \
    1000 \
    "0,1"
```

**Parameters (positional):**
1. `pruned_model_path`: Path to pruned model (required)
2. `workspace`: Output directory (default: `./workspace`)
3. `max_steps`: Training steps (default: `1000`)
4. `devices`: GPU IDs (default: `"0"`)

## Examples

### Quick Start - Iterative Pruning
```bash
# Basic usage with TinyLlama
./scripts/bash/run_iterative_pruning.sh \
    ./checkpoints/tinyllama \
    ./workspace/experiment1 \
    2 \
    10 \
    "0" \
    100

# Multi-GPU with more layers
./scripts/bash/run_iterative_pruning.sh \
    ./checkpoints/llama2-7b \
    ./workspace/llama_experiment \
    5 \
    15 \
    "0,1,2" \
    200
```

### Window Pruning Pipeline
```bash
# Find and prune unimportant window
./scripts/bash/run_window_pruning.sh \
    ./checkpoints/mistral-7b \
    ./workspace/window_exp \
    4 \
    "1"

# Fine-tune the pruned model
./scripts/bash/run_finetune_window_pruned.sh \
    ./workspace/window_exp/window_pruned_model \
    ./workspace/window_exp \
    500 \
    "1"
```

### Individual Scripts
```bash
# Remove specific layers (saves to ./workspace/phi2_pruned automatically)
python scripts/prune_layers.py \
    --model_path ./checkpoints/phi2 \
    --layers 8,16,24 \
    --workspace ./workspace

# Fine-tune with custom parameters
python scripts/finetuning.py \
    --model_path ./models/phi2_pruned \
    --output_path ./models/phi2_finetuned \
    --max_steps 1000 \
    --learning_rate 1e-4 \
    --batch_size 4
```

## Results Analysis

After running experiments, use the standalone notebook to visualize results:

```bash
# Open Jupyter and run
jupyter notebook src/notebooks/results_analysis_standalone.ipynb
```

Or modify the `WORKSPACE` variable in the notebook to point to your experiment directory.
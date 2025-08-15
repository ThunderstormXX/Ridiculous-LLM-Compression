# Attention Head Analysis Examples

## Overview
These scripts analyze attention head similarities and their impact on model perplexity.

## Usage Examples

### 1. Compute Head Similarities
```bash
# Compute similarity matrices for all layers
python compute_head_similarities.py --model_key llama3.1-8b

# This will create:
# - logs/sim_matrices_layer{N}.npy (similarity matrices)
# - figures/sim_layer_{N}.png (heatmaps)
# - logs/sim_matrices_index.json (index file)
```

### 2. Merge Heads and Evaluate
```bash
# Analyze specific layer (e.g., layer 5)
python merge_heads_and_eval.py --model_key llama3.1-8b --layer 5

# With custom sample size
python merge_heads_and_eval.py --model_key llama3.1-8b --layer 5 --max_samples 100

# This will create:
# - logs/perplexities_layer{N}.json (perplexity results)
```

### 3. Plot Results
```bash
# Create visualization of all results
python plot_perplexities.py

# Skip showing plot (just save files)
python plot_perplexities.py --no_plot

# This will create:
# - figures/perplexity_vs_layer.png (plot)
# - logs/perplexities_summary.csv (summary data)
```

## Complete Workflow
```bash
# Step 1: Compute similarities for all layers
python compute_head_similarities.py --model_key llama3.1-8b

# Step 2: Analyze each layer (example for layers 0-5)
for layer in {0..5}; do
    python merge_heads_and_eval.py --model_key llama3.1-8b --layer $layer
done

# Step 3: Create visualization
python plot_perplexities.py
```

## Output Files Structure
```
logs/
├── sim_matrices_layer0.npy
├── sim_matrices_layer1.npy
├── ...
├── sim_matrices_index.json
├── perplexities_layer0.json
├── perplexities_layer1.json
├── ...
└── perplexities_summary.csv

figures/
├── sim_layer_0.png
├── sim_layer_1.png
├── ...
└── perplexity_vs_layer.png
```

## Available Models
- llama3.1-8b
- llama2-13b
- mistral-7b
- phi2
- qwen-7b
- tinyllama
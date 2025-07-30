# PruningHealing Library

A modular library for pruning and healing large language models with LoRA fine-tuning.

## Installation

```bash
pip install -r requirements.txt
```

## Library Structure

- `src/pruninghealing/`: Main library modules
  - `trainer.py`: Training and fine-tuning functionality
  - `dataset.py`: Dataset loading and preparation
  - `prune.py`: Pruning implementations (Iterative and Window)
  - `logger.py`: Experiment logging and visualization
  - `utils.py`: Helper functions

## Scripts

- `scripts/prune_layers.py`: Prune specified decoder layers
- `scripts/finetuning.py`: Fine-tune models on datasets
- `scripts/unimportant_decoder_search.py`: Find least important layers
- `scripts/merge_attention_heads.py`: Merge similar attention heads based on cosine similarity
- `scripts/analyze_head_merging.py`: Analyze head merging results
- `scripts/test_head_merging.py`: Test head merging functionality

## Bash Scripts

- `scripts/bash/run_iterative_pruning.sh`: Run iterative pruning approach
- `scripts/bash/run_window_pruning.sh`: Run window pruning approach
- `scripts/bash/run_finetune_window_pruned.sh`: Fine-tune window-pruned models
- `scripts/bash/run_head_merging.sh`: Run attention head merging experiments

## Usage Examples

### Iterative Pruning
```bash
./scripts/bash/run_iterative_pruning.sh /path/to/model ./workspace 3 0 "0" 50
```

### Window Pruning
```bash
./scripts/bash/run_window_pruning.sh /path/to/model ./workspace 3 "0"
./scripts/bash/run_finetune_window_pruned.sh ./workspace/window_pruned_model ./workspace 1000 "0,1"
```

### Attention Head Merging
```bash
# Run experiments with multiple thresholds
./scripts/bash/run_head_merging.sh --model_path src/checkpoints/llama3.1-8b --thresholds 0.98,0.99,0.995

# Single experiment
python scripts/merge_attention_heads.py --model_path src/checkpoints/llama3.1-8b --threshold 0.99
```

### Python API
```python
from src.pruninghealing import Trainer, DatasetLoader, IterativePruner
from src.pruninghealing.utils import load_model_and_tokenizer

# Load model
model, tokenizer = load_model_and_tokenizer("path/to/model")

# Load dataset
dataset = DatasetLoader(tokenizer).load_wikitext()

# Create trainer
trainer = Trainer(model, tokenizer)

# Train model
trained_model = trainer.train(dataset)
```

## Results Analysis

Use the Jupyter notebook `src/notebooks/results.ipynb` to visualize and compare results from both pruning approaches.

## Supported Architectures

- LLaMA-2
- Mistral
- Phi-2
- Qwen
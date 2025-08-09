# scripts/igor_exps/unified_pruning.py
import argparse
import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.pruninghealing import Trainer, DatasetLoader, IterativePruner, WindowPruner
from src.pruninghealing.prune import DefaultIterativeStrategy, DefaultWindowStrategy
from src.pruninghealing.utils import load_model_and_tokenizer, calculate_perplexity, get_model_layers, safe_save_model
from src.pruninghealing.logger import Logger

def main():
    parser = argparse.ArgumentParser(description="Unified pruning and healing experiment")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--method", choices=["iterative", "window"], required=True, help="Pruning method")
    parser.add_argument("--max_steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    # Method-specific parameters
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers to prune (iterative)")
    parser.add_argument("--start_layer", type=int, default=0, help="Starting layer (iterative)")
    parser.add_argument("--window_size", type=int, default=3, help="Window size (window)")
    
    args = parser.parse_args()
    
    # Create experiment directory
    exp_dir = os.path.join(args.workspace, f"{args.method}_pruning")
    os.makedirs(exp_dir, exist_ok=True)
    run_num = 1
    while os.path.exists(os.path.join(exp_dir, f"run_{run_num}")):
        run_num += 1
    run_dir = os.path.join(exp_dir, f"run_{run_num}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("Loading model...")
    device = args.device
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=device)
    print(f"Model loaded: {get_model_layers(model)} layers")
    
    print("Loading cached dataset...")
    from datasets import load_from_disk
    cached_dataset_path = os.path.join(os.path.dirname(__file__), '../../cached_dataset')
    raw_dataset = load_from_disk(cached_dataset_path)
    
    print("Calculating baseline perplexity...")
    baseline_ppl = calculate_perplexity(model, tokenizer, dataset=raw_dataset['validation'])
    print(f"Baseline perplexity: {baseline_ppl:.3f}")
    
    # Tokenize dataset for training
    print("Tokenizing dataset for training...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset['train'].column_names)
    
    def format_dataset(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_dataset = tokenized_dataset.map(format_dataset, batched=True)
    
    # Create simple dataset object for training
    class SimpleDataset:
        def __init__(self, train_data, eval_data):
            self.train_dataset = train_data
            self.eval_dataset = eval_data
    
    # Initialize components
    trainer = Trainer(model, tokenizer, run_dir)
    logger = Logger(run_dir)
    
    # Log baseline
    logger.log_step({
        "step": 0,
        "action": "baseline",
        "method": args.method,
        "layers_total": get_model_layers(model),
        "perplexity": baseline_ppl
    })
    
    # Run pruning method
    if args.method == "iterative":
        pruner = IterativePruner(model, tokenizer, run_dir)
        search_strategy = DefaultIterativeStrategy()
        dataset_obj = SimpleDataset(tokenized_dataset['train'], raw_dataset['validation'])
        final_model = pruner.prune_and_heal(
            dataset=dataset_obj,
            trainer=trainer,
            logger=logger,
            start_layer=args.start_layer,
            num_layers=args.num_layers,
            max_steps=args.max_steps,
            search_strategy=search_strategy
        )
        method_suffix = "iter"
    else:  # window
        pruner = WindowPruner(model, tokenizer, run_dir)
        search_strategy = DefaultWindowStrategy()
        dataset_obj = SimpleDataset(tokenized_dataset['train'], raw_dataset['validation'])
        final_model = pruner.prune_and_heal(
            dataset=dataset_obj,
            trainer=trainer,
            logger=logger,
            window_size=args.window_size,
            max_steps=args.max_steps,
            search_strategy=search_strategy
        )
        method_suffix = "window"
    
    # Save final model
    model_name = os.path.basename(args.model_path.rstrip('/'))
    output_path = f"src/checkpoints/{model_name}_p_{method_suffix}"
    os.makedirs("src/checkpoints", exist_ok=True)
    
    print(f"\nSaving final model to: {output_path}")
    safe_save_model(final_model, output_path)
    tokenizer.save_pretrained(output_path)
    
    print("\nExperiment completed!")
    print(f"Results saved to: {run_dir}")
    print(f"Final model saved to: {output_path}")

if __name__ == "__main__":
    main()
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
    
    # Load model and tokenizer
    print("Loading model...")
    # Use cuda:0 instead of auto to avoid device_map issues with PEFT
    device = "cuda:0" if args.device == "auto" else args.device
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=device)
    print(f"Model loaded: {get_model_layers(model)} layers")
    
    # Load dataset
    print("Loading C4 dataset...")
    dataset_loader = DatasetLoader(tokenizer)
    dataset_loader.load_c4(train_size=500, eval_size=50)
    
    # Calculate baseline perplexity
    print("Calculating baseline perplexity...")
    baseline_ppl = calculate_perplexity(model, tokenizer, dataset=dataset_loader.eval_dataset, max_samples=20)
    print(f"Baseline perplexity: {baseline_ppl:.3f}")
    
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
        final_model = pruner.prune_and_heal(
            dataset=dataset_loader,
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
        final_model = pruner.prune_and_heal(
            dataset=dataset_loader,
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
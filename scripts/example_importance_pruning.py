#!/usr/bin/env python3
"""
Example script showing how to use importance-based pruning strategies
"""
import argparse
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pruninghealing import (
    IterativePruner, WindowPruner, 
    ImportanceBasedIterativeStrategy, ImportanceBasedWindowStrategy,
    compute_layer_importances, Trainer, DatasetLoader, Logger
)
from src.pruninghealing.utils import load_model_and_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Example of importance-based pruning")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--hidden_path", type=str, required=True, help="Path to hidden states JSON")
    parser.add_argument("--pruner_type", type=str, choices=["iterative", "window"], default="iterative",
                       help="Type of pruner to use")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers to prune")
    parser.add_argument("--workspace", type=str, default="./workspace", help="Workspace directory")
    parser.add_argument("--layer_type", type=str, default="mlp", help="Layer type for importance calculation")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--skip_training", action="store_true", help="Skip training (debug mode)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    # Compute layer importances
    print(f"Computing layer importances from {args.hidden_path}...")
    importances = compute_layer_importances(args.hidden_path, args.layer_type)
    
    # Create importance-based strategy
    if args.pruner_type == "iterative":
        strategy = ImportanceBasedIterativeStrategy(importances)
        pruner = IterativePruner(model, tokenizer, args.workspace)
    else:
        strategy = ImportanceBasedWindowStrategy(importances)
        pruner = WindowPruner(model, tokenizer, args.workspace)
    
    # Load dataset
    print("Loading dataset...")
    dataset_loader = DatasetLoader(tokenizer)
    dataset = dataset_loader.load_wikitext()
    
    # Create trainer and logger
    trainer = Trainer(model, tokenizer)
    logger = Logger(args.workspace)
    
    # Run pruning with importance-based strategy
    print(f"Starting {args.pruner_type} pruning with importance-based strategy...")
    if args.pruner_type == "iterative":
        pruned_model = pruner.prune_and_heal(
            dataset, trainer, logger,
            num_layers=args.num_layers,
            max_steps=args.max_steps,
            search_strategy=strategy,
            skip_training=args.skip_training
        )
    else:
        pruned_model = pruner.prune_and_heal(
            dataset, trainer, logger,
            window_size=args.num_layers,
            max_steps=args.max_steps,
            search_strategy=strategy,
            skip_training=args.skip_training
        )
    
    print("Pruning completed!")
    
    # Save final model
    output_path = os.path.join(args.workspace, f"{args.pruner_type}_importance_pruned")
    pruned_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()
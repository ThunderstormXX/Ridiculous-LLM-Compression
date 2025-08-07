# scripts/igor_exps/prune_layers.py
import argparse
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.pruninghealing.utils import load_model_and_tokenizer, get_model_layers
from src.pruninghealing.prune import IterativePruner

def main():
    parser = argparse.ArgumentParser(description="Prune specified decoder layers from model")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices to remove")
    parser.add_argument("--output_path", help="Path to save pruned model (default: model_path + '_pruned')")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output_path is None:
        import os
        model_name = os.path.basename(args.model_path.rstrip('/'))
        args.output_path = f"src/checkpoints/{model_name}_p_window"
    
    os.makedirs(args.workspace, exist_ok=True)
    os.makedirs("src/checkpoints", exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    # Parse layer indices
    layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    
    print(f"Original model layers: {get_model_layers(model)}")
    print(f"Removing layers: {layer_indices}")
    
    # Create pruner and remove layers
    pruner = IterativePruner(model, tokenizer, args.workspace)
    
    # Remove specified layers
    for layer_idx in sorted(layer_indices, reverse=True):  # Remove from end to start
        model = pruner._remove_layer(model, layer_idx)
    
    print(f"Remaining layers: {get_model_layers(model)}")
    
    # Save pruned model
    from src.pruninghealing.utils import safe_save_model
    safe_save_model(model, args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    print(f"Pruned model saved to: {args.output_path}")

if __name__ == "__main__":
    main()
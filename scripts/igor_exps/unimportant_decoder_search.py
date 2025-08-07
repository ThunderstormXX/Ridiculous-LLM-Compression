# scripts/igor_exps/unimportant_decoder_search.py
import argparse
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.pruninghealing.utils import load_model_and_tokenizer, calculate_perplexity
from src.pruninghealing.prune import WindowPruner

def main():
    parser = argparse.ArgumentParser(description="Find least important decoder layers/windows")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--window_size", type=int, default=3, help="Size of layer window to evaluate")
    parser.add_argument("--output_file", default="unimportant_layers.json", help="Output file for results")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    # Create window pruner
    pruner = WindowPruner(model, tokenizer, args.workspace)
    
    # Find unimportant window
    print(f"Searching for least important window of size {args.window_size}...")
    best_window, best_score = pruner.find_unimportant_window(args.window_size)
    
    # Calculate baseline perplexity
    baseline_ppl = calculate_perplexity(model, tokenizer)
    
    results = {
        "baseline_perplexity": baseline_ppl,
        "best_window": best_window,
        "best_score": best_score,
        "window_size": args.window_size,
        "model_path": args.model_path
    }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {args.output_file}")
    print(f"Least important window: layers {best_window}")
    print(f"Baseline perplexity: {baseline_ppl:.3f}")

if __name__ == "__main__":
    main()
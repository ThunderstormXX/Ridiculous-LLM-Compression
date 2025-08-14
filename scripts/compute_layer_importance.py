#!/usr/bin/env python3
"""
Script to compute layer importance from hidden states
"""
import argparse
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pruninghealing.layer_importance import compute_layer_importances

def main():
    parser = argparse.ArgumentParser(description="Compute layer importance from hidden states")
    parser.add_argument("--hidden_path", type=str, required=True,
                       help="Path to json with hiddens")
    parser.add_argument("--layer_type", type=str, default="mlp",
                       help="Layer type to measure: 'input_layernorm', 'self_attn', 'post_attention_layernorm', 'mlp'")
    parser.add_argument("--output", type=str, default="layer_importances.json",
                       help="Output file for importances")
    
    args = parser.parse_args()
    
    # Compute importances
    importances = compute_layer_importances(args.hidden_path, args.layer_type)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(importances, f, indent=2)
    
    print(f"Layer importances computed and saved to {args.output}")
    print("Importances (sorted by importance):")
    for layer_idx, importance in importances:
        print(f"  Layer {layer_idx}: {importance:.6f}")

if __name__ == "__main__":
    main()
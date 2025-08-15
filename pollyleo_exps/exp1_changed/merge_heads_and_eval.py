#!/usr/bin/env python3
"""
Merge attention heads and evaluate perplexity impact.

Usage:
    python merge_heads_and_eval.py --model_key llama3.1-8b --layer 5

This script:
1. Loads a model and its similarity matrix for the specified layer
2. Finds the most similar and most dissimilar head pairs
3. Merges heads by averaging their weights in q_proj, k_proj, v_proj, o_proj
4. Evaluates normalized perplexity before and after merging
5. Saves results to JSON log file
"""

#!/usr/bin/env python3
"""
Merge attention heads and evaluate perplexity impact with minimal memory usage.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.pruninghealing.utils import calculate_perplexity

CHECKPOINTS_DIR = Path("../../src/checkpoints")
LOGS_DIR = Path("logs")
CACHED_DATASET_PATH = Path("../../cached_dataset")

def load_saved_model(model_key):
    model_path = CHECKPOINTS_DIR / model_key
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_key} not found in {CHECKPOINTS_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    return model, tokenizer

def load_cached_dataset():
    """Load cached dataset"""
    return load_from_disk(CACHED_DATASET_PATH)

def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    else:
        raise ValueError("Cannot find model layers")

def merge_heads(linear_layer, head_a, head_b, num_heads):
    head_dim = linear_layer.weight.shape[0] // num_heads
    model_dim = linear_layer.weight.shape[1]

    with torch.no_grad():
        w = linear_layer.weight.detach().clone().view(num_heads, head_dim, model_dim)
        avg = 0.5 * (w[head_a] + w[head_b])
        w[head_a] = avg
        w[head_b] = avg
        linear_layer.weight.data.copy_(w.view(num_heads * head_dim, model_dim))

def find_head_pairs(similarity_matrix):
    num_heads = similarity_matrix.shape[0]
    mask = np.ones_like(similarity_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    
    max_sim_idx = np.unravel_index(np.argmax(similarity_matrix * mask), similarity_matrix.shape)
    max_similarity = similarity_matrix[max_sim_idx]
    
    min_sim_idx = np.unravel_index(np.argmin(similarity_matrix * mask), similarity_matrix.shape)
    min_similarity = similarity_matrix[min_sim_idx]
    
    return max_sim_idx, max_similarity, min_sim_idx, min_similarity

def clone_layer_weights(layer):
    """Клонируем только веса self_attn q,k,v,o_proj"""
    from copy import deepcopy
    layer_clone = deepcopy(layer)
    
    # Check available attributes in self_attn
    attn = layer.self_attn
    proj_names = []
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if hasattr(attn, name):
            proj_names.append(name)
    
    # If standard names don't exist, try alternative names
    if not proj_names:
        alt_names = ['query', 'key', 'value', 'dense']
        for name in alt_names:
            if hasattr(attn, name):
                proj_names.append(name)
    
    for proj_name in proj_names:
        orig_proj = getattr(attn, proj_name)
        new_proj = getattr(layer_clone.self_attn, proj_name)
        new_proj.weight.data = orig_proj.weight.data.clone()
        if hasattr(orig_proj, 'bias') and orig_proj.bias is not None:
            new_proj.bias.data = orig_proj.bias.data.clone()
    
    return layer_clone

def main():
    parser = argparse.ArgumentParser(description="Merge attention heads and evaluate perplexity")
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()
    
    print(f"Analyzing layer {args.layer} of model {args.model_key}")
    
    sim_matrix_path = LOGS_DIR / f"sim_matrices_layer{args.layer}.npy"
    if not sim_matrix_path.exists():
        raise FileNotFoundError(f"Similarity matrix not found: {sim_matrix_path}")
    similarity_matrix = np.load(sim_matrix_path)
    
    similar_pair, max_sim, dissimilar_pair, min_sim = find_head_pairs(similarity_matrix)
    
    model, tokenizer = load_saved_model(args.model_key)
    layers = get_model_layers(model)
    target_layer = layers[args.layer]
    num_heads = model.config.num_attention_heads
    
    # Load cached dataset
    print("Loading cached dataset...")
    cached_dataset = load_cached_dataset()
    eval_dataset = cached_dataset['validation']
    
    print("Calculating baseline perplexity...")
    baseline_perplexity = calculate_perplexity(model, tokenizer, eval_dataset, max_samples=args.max_samples, normalized=True)
    print(f"Baseline normalized perplexity: {baseline_perplexity:.4f}")
    
    # --- Мердж похожих голов ---
    print("Testing similar heads merging...")
    layer_similar = clone_layer_weights(target_layer)
    
    # Find available projection layers
    attn = layer_similar.self_attn
    proj_names = []
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if hasattr(attn, name):
            proj_names.append(name)
    
    for proj in proj_names:
        merge_heads(getattr(attn, proj), similar_pair[0], similar_pair[1], num_heads)
    
    # Вставляем изменённый слой в модель (временно)
    with torch.no_grad():
        orig_layer = layers[args.layer]
        layers[args.layer] = layer_similar
        similar_perplexity = calculate_perplexity(model, tokenizer, eval_dataset, max_samples=args.max_samples, normalized=True)
        layers[args.layer] = orig_layer
    
    torch.cuda.empty_cache()
    print(f"After merging similar heads: {similar_perplexity:.4f}")
    
    # --- Мердж разных голов ---
    print("Testing dissimilar heads merging...")
    layer_dissimilar = clone_layer_weights(target_layer)
    
    attn_dissimilar = layer_dissimilar.self_attn
    for proj in proj_names:  # Use same proj_names from above
        merge_heads(getattr(attn_dissimilar, proj), dissimilar_pair[0], dissimilar_pair[1], num_heads)
    
    with torch.no_grad():
        orig_layer = layers[args.layer]
        layers[args.layer] = layer_dissimilar
        dissimilar_perplexity = calculate_perplexity(model, tokenizer, eval_dataset, max_samples=args.max_samples, normalized=True)
        layers[args.layer] = orig_layer
    
    torch.cuda.empty_cache()
    print(f"After merging dissimilar heads: {dissimilar_perplexity:.4f}")
    
    results = {
        "layer": args.layer,
        "similar_heads": [int(similar_pair[0]), int(similar_pair[1])],
        "dissimilar_heads": [int(dissimilar_pair[0]), int(dissimilar_pair[1])],
        "max_similarity": float(max_sim),
        "min_similarity": float(min_sim),
        "before": float(baseline_perplexity),
        "after_similar": float(similar_perplexity),
        "after_dissimilar": float(dissimilar_perplexity)
    }
    
    results_path = LOGS_DIR / f"perplexities_layer{args.layer}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    print("Analysis completed!")

if __name__ == "__main__":
    main()

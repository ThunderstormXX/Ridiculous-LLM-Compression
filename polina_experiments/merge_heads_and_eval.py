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

import argparse
import json
import math
import copy
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Setup paths
CHECKPOINTS_DIR = Path("../src/checkpoints")
LOGS_DIR = Path("logs")

def load_saved_model(model_key):
    """Load model and tokenizer from checkpoints directory"""
    model_path = CHECKPOINTS_DIR / model_key
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_key} not found in {CHECKPOINTS_DIR}")
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def calculate_perplexity(model, tokenizer, dataset=None, dataset_name="allenai/c4", dataset_config="en", max_samples=100):
    """Calculate normalized perplexity on evaluation dataset"""
    model.eval()
    device = next(model.parameters()).device
    
    if dataset is None:
        if dataset_name == "allenai/c4":
            eval_dataset = load_dataset(dataset_name, dataset_config, split="validation", streaming=True)
            eval_samples = []
            for i, sample in enumerate(eval_dataset):
                if len(eval_samples) >= max_samples:
                    break
                eval_samples.append(sample)
            from datasets import Dataset
            eval_dataset = Dataset.from_list(eval_samples)
        else:
            eval_dataset = load_dataset(dataset_name, dataset_config, split="validation")
            eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    else:
        eval_dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for example in eval_dataset:
            text = example.get("text", None)
            if text is None or not text.strip():
                continue
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Normalize by vocabulary size
    vocab_size = model.config.vocab_size
    norm_perplexity = perplexity / math.log(vocab_size)
    
    model.train()
    return norm_perplexity

def get_model_layers(model):
    """Get model layers - works for LLaMA, Mistral, etc."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    else:
        raise ValueError("Cannot find model layers")

def merge_heads(linear_layer, head_a, head_b, num_heads):
    with torch.no_grad():
        w = linear_layer.weight.detach().clone()
        out_dim, in_dim = w.shape

        head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "Output dimension not divisible by number of heads"

        # reshape to (num_heads, head_dim, in_dim)
        w = w.view(num_heads, head_dim, in_dim)

        # average the two heads
        avg = 0.5 * (w[head_a] + w[head_b])
        w[head_a] = avg
        w[head_b] = avg

        # copy back
        linear_layer.weight.data.copy_(w.view_as(linear_layer.weight))


def find_head_pairs(similarity_matrix):
    """Find most similar and most dissimilar head pairs"""
    num_heads = similarity_matrix.shape[0]
    
    # Find most similar pair (excluding diagonal)
    mask = np.ones_like(similarity_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    
    max_sim_idx = np.unravel_index(np.argmax(similarity_matrix * mask), similarity_matrix.shape)
    max_similarity = similarity_matrix[max_sim_idx]
    
    # Find most dissimilar pair
    min_sim_idx = np.unravel_index(np.argmin(similarity_matrix * mask), similarity_matrix.shape)
    min_similarity = similarity_matrix[min_sim_idx]
    
    return max_sim_idx, max_similarity, min_sim_idx, min_similarity

def main():
    parser = argparse.ArgumentParser(description="Merge attention heads and evaluate perplexity")
    parser.add_argument("--model_key", required=True, help="Model key from checkpoints directory")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to analyze")
    parser.add_argument("--max_samples", type=int, default=50, help="Number of samples for perplexity evaluation")
    args = parser.parse_args()
    
    print(f"Analyzing layer {args.layer} of model {args.model_key}")
    
    # Load similarity matrix
    sim_matrix_path = LOGS_DIR / f"sim_matrices_layer{args.layer}.npy"
    if not sim_matrix_path.exists():
        raise FileNotFoundError(f"Similarity matrix not found: {sim_matrix_path}")
    
    similarity_matrix = np.load(sim_matrix_path)
    print(f"Loaded similarity matrix: {similarity_matrix.shape}")
    
    # Find head pairs
    similar_pair, max_sim, dissimilar_pair, min_sim = find_head_pairs(similarity_matrix)
    print(f"Most similar heads: {similar_pair} (similarity: {max_sim:.4f})")
    print(f"Most dissimilar heads: {dissimilar_pair} (similarity: {min_sim:.4f})")
    
    # Load model
    model, tokenizer = load_saved_model(args.model_key)
    layers = get_model_layers(model)
    target_layer = layers[args.layer]
    num_heads = model.config.num_attention_heads
    
    # Calculate baseline perplexity
    print("Calculating baseline perplexity...")
    baseline_perplexity = calculate_perplexity(model, tokenizer, max_samples=args.max_samples)
    print(f"Baseline normalized perplexity: {baseline_perplexity:.4f}")
    
    # Test similar heads merging
    print("Testing similar heads merging...")
    model_copy = copy.deepcopy(model)
    layers_copy = get_model_layers(model_copy)
    layer_copy = layers_copy[args.layer]
    
    # Merge similar heads
    merge_heads(layer_copy.self_attn.q_proj, similar_pair[0], similar_pair[1], num_heads)
    merge_heads(layer_copy.self_attn.k_proj, similar_pair[0], similar_pair[1], num_heads)
    merge_heads(layer_copy.self_attn.v_proj, similar_pair[0], similar_pair[1], num_heads)
    merge_heads(layer_copy.self_attn.o_proj, similar_pair[0], similar_pair[1], num_heads)
    
    similar_perplexity = calculate_perplexity(model_copy, tokenizer, max_samples=args.max_samples)
    print(f"After merging similar heads: {similar_perplexity:.4f}")
    
    # Test dissimilar heads merging
    print("Testing dissimilar heads merging...")
    model_copy2 = copy.deepcopy(model)
    layers_copy2 = get_model_layers(model_copy2)
    layer_copy2 = layers_copy2[args.layer]
    
    # Merge dissimilar heads
    merge_heads(layer_copy2.self_attn.q_proj, dissimilar_pair[0], dissimilar_pair[1], num_heads)
    merge_heads(layer_copy2.self_attn.k_proj, dissimilar_pair[0], dissimilar_pair[1], num_heads)
    merge_heads(layer_copy2.self_attn.v_proj, dissimilar_pair[0], dissimilar_pair[1], num_heads)
    merge_heads(layer_copy2.self_attn.o_proj, dissimilar_pair[0], dissimilar_pair[1], num_heads)
    
    dissimilar_perplexity = calculate_perplexity(model_copy2, tokenizer, max_samples=args.max_samples)
    print(f"After merging dissimilar heads: {dissimilar_perplexity:.4f}")
    
    # Save results
    results = {
        "layer": args.layer,
        "similar_heads": similar_pair,
        "dissimilar_heads": dissimilar_pair,
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
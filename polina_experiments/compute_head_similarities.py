#!/usr/bin/env python3
"""
Compute attention head similarities for each layer of a language model.

Usage:
    python compute_head_similarities.py --model_key llama3.1-8b

This script:
1. Loads a saved model using the provided model key
2. For each layer, extracts Q and K projection weights
3. Computes similarity matrices between attention heads using cosine similarity
4. Saves matrices as .npy files and visualizations as heatmaps
5. Creates an index JSON file for easy access
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup paths
CHECKPOINTS_DIR = Path("../src/checkpoints")
LOGS_DIR = Path("logs")
FIGURES_DIR = Path("figures")

LOGS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

def load_saved_model(model_key):
    """Load model and tokenizer from checkpoints directory"""
    model_path = CHECKPOINTS_DIR / model_key
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_key} not found in {CHECKPOINTS_DIR}")
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom

def get_model_layers(model):
    """Get model layers - works for LLaMA, Mistral, etc."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    else:
        raise ValueError("Cannot find model layers")

def compute_head_similarities(model, layer_idx):
    """Compute similarity matrix for attention heads in a given layer"""
    layers = get_model_layers(model)
    layer = layers[layer_idx]
    
    # Get attention weights
    q_proj = layer.self_attn.q_proj.weight.data.cpu().numpy()
    k_proj = layer.self_attn.k_proj.weight.data.cpu().numpy()
    
    num_heads = model.config.num_attention_heads
    head_dim = q_proj.shape[0] // num_heads
    
    # Reshape to separate heads
    q_heads = q_proj.reshape(num_heads, head_dim, -1)
    k_heads = k_proj.reshape(num_heads, head_dim, -1)
    
    # Compute M_h = Q_h @ K_h.T for each head and vectorize
    head_matrices = []
    for h in range(num_heads):
        M_h = q_heads[h] @ k_heads[h].T
        head_matrices.append(M_h.flatten())
    
    # Compute similarity matrix
    similarity_matrix = np.zeros((num_heads, num_heads))
    for i in range(num_heads):
        for j in range(num_heads):
            similarity_matrix[i, j] = cosine_similarity(head_matrices[i], head_matrices[j])
    
    return similarity_matrix

def save_similarity_matrix(similarity_matrix, layer_idx):
    """Save similarity matrix as .npy file and create heatmap"""
    # Save matrix
    matrix_path = LOGS_DIR / f"sim_matrices_layer{layer_idx}.npy"
    np.save(matrix_path, similarity_matrix)
    print(f"Saved similarity matrix for layer {layer_idx} to {matrix_path}")
    
    # Create and save heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'label': 'Cosine Similarity'})
    plt.title(f'Attention Head Similarity Matrix - Layer {layer_idx}')
    plt.xlabel('Head Index')
    plt.ylabel('Head Index')
    
    heatmap_path = FIGURES_DIR / f"sim_layer_{layer_idx}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap for layer {layer_idx} to {heatmap_path}")

def create_index_file(num_layers):
    """Create index JSON file with metadata"""
    index_data = {
        "num_layers": num_layers,
        "files": [f"sim_matrices_layer{i}.npy" for i in range(num_layers)],
        "heatmaps": [f"sim_layer_{i}.png" for i in range(num_layers)]
    }
    
    index_path = LOGS_DIR / "sim_matrices_index.json"
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    print(f"Created index file: {index_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute attention head similarities")
    parser.add_argument("--model_key", required=True, help="Model key from checkpoints directory")
    args = parser.parse_args()
    
    print(f"Computing head similarities for model: {args.model_key}")
    
    # Load model
    model, tokenizer = load_saved_model(args.model_key)
    
    # Get number of layers
    layers = get_model_layers(model)
    num_layers = len(layers)
    print(f"Model has {num_layers} layers")
    
    # Process each layer
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers-1}...")
        similarity_matrix = compute_head_similarities(model, layer_idx)
        save_similarity_matrix(similarity_matrix, layer_idx)
    
    # Create index file
    create_index_file(num_layers)
    
    print("Head similarity computation completed!")

if __name__ == "__main__":
    main()
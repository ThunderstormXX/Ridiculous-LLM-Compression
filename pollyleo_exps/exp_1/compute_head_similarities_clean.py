#!/usr/bin/env python3
"""
Compute attention head similarities for each layer of a language model.
Only saves matrices, no visualization.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup paths
CHECKPOINTS_DIR = Path("../../src/checkpoints")
LOGS_DIR = Path("logs")

LOGS_DIR.mkdir(exist_ok=True)

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

def get_model_layers(model):
    """Get model layers - works for LLaMA, Mistral, etc."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    else:
        raise ValueError("Cannot find model layers")

def compute_head_similarities(model, layer_idx):
    layer = model.model.layers[layer_idx]
    q_proj = layer.self_attn.q_proj.weight.detach().cpu()
    k_proj = layer.self_attn.k_proj.weight.detach().cpu()

    num_q_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)

    head_dim = q_proj.shape[0] // num_q_heads

    q_heads = q_proj.view(num_q_heads, head_dim, -1)
    k_heads = k_proj.view(num_kv_heads, head_dim, -1)

    if num_kv_heads != num_q_heads:
        repeat_factor = num_q_heads // num_kv_heads
        k_heads = k_heads.repeat_interleave(repeat_factor, dim=0)

    similarity_matrix = torch.zeros((num_q_heads, num_q_heads))
    for i in range(num_q_heads):
        for j in range(num_q_heads):
            qi_ki = torch.matmul(q_heads[i], k_heads[i].T)
            qj_kj = torch.matmul(q_heads[j], k_heads[j].T)

            cos_sim = torch.nn.functional.cosine_similarity(
                qi_ki.flatten(), qj_kj.flatten(), dim=0
            )
            similarity_matrix[i, j] = cos_sim

    return similarity_matrix

def save_similarity_matrix(similarity_matrix, layer_idx):
    """Save similarity matrix as .npy file"""
    matrix_path = LOGS_DIR / f"sim_matrices_layer{layer_idx}.npy"
    np.save(matrix_path, similarity_matrix)
    print(f"Saved similarity matrix for layer {layer_idx} to {matrix_path}")

def create_index_file(num_layers):
    """Create index JSON file with metadata"""
    index_data = {
        "num_layers": num_layers,
        "files": [f"sim_matrices_layer{i}.npy" for i in range(num_layers)]
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
    
    model, tokenizer = load_saved_model(args.model_key)
    
    layers = get_model_layers(model)
    num_layers = len(layers)
    print(f"Model has {num_layers} layers")
    
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers-1}...")
        similarity_matrix = compute_head_similarities(model, layer_idx)
        save_similarity_matrix(similarity_matrix, layer_idx)
    
    create_index_file(num_layers)
    
    print("Head similarity computation completed!")

if __name__ == "__main__":
    main()
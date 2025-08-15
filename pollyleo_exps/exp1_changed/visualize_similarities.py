#!/usr/bin/env python3
"""
Create visualizations from saved similarity matrices.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

LOGS_DIR = Path("logs")
FIGURES_DIR = Path("figures")

FIGURES_DIR.mkdir(exist_ok=True)

def load_similarity_matrices():
    """Load all saved similarity matrices"""
    index_path = LOGS_DIR / "sim_matrices_index.json"
    
    if not index_path.exists():
        raise FileNotFoundError("Index file not found. Run compute_head_similarities_clean.py first.")
    
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    matrices = {}
    for i in range(index_data["num_layers"]):
        matrix_path = LOGS_DIR / f"sim_matrices_layer{i}.npy"
        if matrix_path.exists():
            matrices[i] = np.load(matrix_path)
    
    return matrices

def create_heatmap(similarity_matrix, layer_idx):
    """Create improved heatmap with better formatting"""
    plt.figure(figsize=(14, 12))
    
    # Улучшенное форматирование для избежания слипания цифр
    ax = sns.heatmap(
        similarity_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0,
        square=True, 
        fmt='.2f',  # Меньше знаков после запятой
        annot_kws={'size': 8, 'weight': 'bold'},  # Размер и жирность шрифта
        cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8},
        linewidths=0.5,  # Линии между ячейками
        linecolor='white'
    )
    
    plt.title(f'Attention Head Similarity Matrix - Layer {layer_idx}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Head Index', fontsize=14, fontweight='bold')
    plt.ylabel('Head Index', fontsize=14, fontweight='bold')
    
    # Улучшенные тики
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    heatmap_path = FIGURES_DIR / f"sim_layer_{layer_idx}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved heatmap for layer {layer_idx} to {heatmap_path}")

def main():
    print("Loading similarity matrices...")
    matrices = load_similarity_matrices()
    
    print(f"Creating visualizations for {len(matrices)} layers...")
    for layer_idx, matrix in matrices.items():
        create_heatmap(matrix, layer_idx)
    
    print("Visualization completed!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Анализ схожести голов внимания с использованием L1 метрики.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def load_attention_matrices():
    """Загрузка сохраненных матриц внимания"""
    attention_dir = Path("attentions")
    
    files = list(attention_dir.glob("attention_layer*_head*.pt"))
    layers = set()
    heads_per_layer = set()
    
    for file in files:
        parts = file.stem.split("_")
        layer_idx = int(parts[1].replace("layer", ""))
        head_idx = int(parts[2].replace("head", ""))
        layers.add(layer_idx)
        heads_per_layer.add(head_idx)
    
    num_layers = max(layers) + 1
    num_heads = max(heads_per_layer) + 1
    
    attention_matrices = {}
    for layer_idx in range(num_layers):
        attention_matrices[layer_idx] = {}
        for head_idx in range(num_heads):
            filename = f"attention_layer{layer_idx}_head{head_idx}.pt"
            filepath = attention_dir / filename
            if filepath.exists():
                attention_matrices[layer_idx][head_idx] = torch.load(filepath)
    
    return attention_matrices, num_layers, num_heads

def compute_l1_differences(attention_matrices, num_layers, num_heads):
    """Вычисление L1 разностей между парами голов"""
    layer_differences = {}
    
    for layer_idx in range(num_layers):
        print(f"Обработка слоя {layer_idx}")
        differences = {}
        
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                if i in attention_matrices[layer_idx] and j in attention_matrices[layer_idx]:
                    matrix_i = attention_matrices[layer_idx][i]
                    matrix_j = attention_matrices[layer_idx][j]
                    
                    l1_diff = torch.mean(torch.abs(matrix_i - matrix_j)).item()
                    differences[(i, j)] = l1_diff
        
        layer_differences[layer_idx] = differences
    
    return layer_differences

def find_top_similar_pairs(layer_differences, top_percent=0.1):
    """Поиск топ 10% наиболее похожих пар"""
    all_pairs = []
    
    for layer_idx, differences in layer_differences.items():
        for (i, j), diff in differences.items():
            all_pairs.append((layer_idx, i, j, diff))
    
    all_pairs.sort(key=lambda x: x[3])
    
    top_count = max(1, int(len(all_pairs) * top_percent))
    top_pairs = all_pairs[:top_count]
    
    return top_pairs

def save_similar_heads(top_pairs):
    """Сохранение результатов в файл"""
    output_file = Path("similar_heads.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Топ 10% наиболее похожих пар голов внимания:\n\n")
        
        for layer_idx, head_i, head_j, diff in top_pairs:
            f.write(f"Слой {layer_idx}: Голова {head_i} — Голова {head_j} | Средняя_разница: {diff:.6f}\n")
    
    print(f"Результаты сохранены в {output_file}")

def create_heatmaps(attention_matrices, top_pairs):
    """Создание тепловых карт для похожих пар"""
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    for layer_idx, head_i, head_j, diff in top_pairs:
        if layer_idx in attention_matrices:
            if head_i in attention_matrices[layer_idx] and head_j in attention_matrices[layer_idx]:
                matrix_i = attention_matrices[layer_idx][head_i].numpy()
                matrix_j = attention_matrices[layer_idx][head_j].numpy()
                
                diff_matrix = np.abs(matrix_i - matrix_j)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(diff_matrix, cmap='viridis', cbar_kws={'label': 'L1 Difference'})
                plt.title(f'Слой {layer_idx}: Голова {head_i} vs Голова {head_j}\nСредняя разница: {diff:.6f}')
                plt.xlabel('Позиция токена')
                plt.ylabel('Позиция токена')
                
                filename = f"layer{layer_idx}_head{head_i}_head{head_j}.png"
                plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
    
    print(f"Тепловые карты сохранены в {plots_dir}")

def create_distribution_plots(layer_differences):
    """Создание графиков распределения L1 разностей по слоям"""
    plots_dir = Path("plots")
    
    for layer_idx, differences in layer_differences.items():
        if differences:
            values = list(differences.values())
            
            plt.figure(figsize=(10, 6))
            plt.hist(values, bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Распределение L1 разностей - Слой {layer_idx}')
            plt.xlabel('L1 разность')
            plt.ylabel('Частота')
            plt.grid(True, alpha=0.3)
            
            filename = f"layer{layer_idx}_distribution.png"
            plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    print("Графики распределения сохранены")

def main():
    print("Загрузка матриц внимания...")
    attention_matrices, num_layers, num_heads = load_attention_matrices()
    
    print("Вычисление L1 разностей...")
    layer_differences = compute_l1_differences(attention_matrices, num_layers, num_heads)
    
    print("Поиск наиболее похожих пар...")
    top_pairs = find_top_similar_pairs(layer_differences)
    
    print("Сохранение результатов...")
    save_similar_heads(top_pairs)
    
    print("Создание тепловых карт...")
    create_heatmaps(attention_matrices, top_pairs)
    
    print("Создание графиков распределения...")
    create_distribution_plots(layer_differences)
    
    print("Анализ завершен!")

if __name__ == "__main__":
    main()
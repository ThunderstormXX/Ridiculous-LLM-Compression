#!/usr/bin/env python3
"""
Анализ результатов объединения attention-голов с разными порогами схожести.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

def load_results(results_dir):
    """Загрузить результаты для всех порогов"""
    results = {}
    
    for file_path in Path(results_dir).glob("results_threshold_*.json"):
        threshold = float(file_path.stem.split('_')[-1])
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            results[threshold] = data
    
    return results

def create_threshold_comparison(results, output_dir):
    """Создать график сравнения для разных порогов"""
    thresholds = sorted(results.keys())
    
    # Извлекаем метрики
    original_perplexity = results[thresholds[0]]['original_metrics']['perplexity']
    original_accuracy = results[thresholds[0]]['original_metrics']['accuracy']
    
    merged_perplexities = [results[t]['merged_metrics']['perplexity'] for t in thresholds]
    merged_accuracies = [results[t]['merged_metrics']['accuracy'] for t in thresholds]
    total_pairs = [results[t]['total_merged_pairs'] for t in thresholds]
    
    # Создаем график
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Perplexity vs threshold
    ax1.plot(thresholds, [original_perplexity] * len(thresholds), 'b--', 
             label='Original', linewidth=2, alpha=0.7)
    ax1.plot(thresholds, merged_perplexities, 'ro-', label='After merging', linewidth=2)
    ax1.set_xlabel('Similarity Threshold')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity vs Similarity Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy vs threshold
    ax2.plot(thresholds, [original_accuracy] * len(thresholds), 'b--', 
             label='Original', linewidth=2, alpha=0.7)
    ax2.plot(thresholds, merged_accuracies, 'go-', label='After merging', linewidth=2)
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Similarity Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Number of merged pairs vs threshold
    ax3.bar(thresholds, total_pairs, alpha=0.7, color='orange', edgecolor='darkorange')
    ax3.set_xlabel('Similarity Threshold')
    ax3.set_ylabel('Total Merged Pairs')
    ax3.set_title('Number of Merged Pairs vs Threshold')
    ax3.grid(True, alpha=0.3)
    
    # Relative performance change
    perp_change = [(p - original_perplexity) / original_perplexity * 100 
                   for p in merged_perplexities]
    acc_change = [(a - original_accuracy) / original_accuracy * 100 
                  for a in merged_accuracies]
    
    ax4.plot(thresholds, perp_change, 'ro-', label='Perplexity change (%)', linewidth=2)
    ax4.plot(thresholds, acc_change, 'go-', label='Accuracy change (%)', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Similarity Threshold')
    ax4.set_ylabel('Relative Change (%)')
    ax4.set_title('Relative Performance Change')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_layer_analysis(results, output_dir):
    """Создать анализ по слоям для каждого порога"""
    thresholds = sorted(results.keys())
    
    for threshold in thresholds:
        data = results[threshold]
        layer_results = data['layer_results']
        
        if not layer_results:
            continue
            
        layers = sorted([int(k) for k in layer_results.keys()])
        num_pairs = [layer_results[str(layer)]['num_pairs'] for layer in layers]
        
        # Создаем heatmap схожести для каждого слоя
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Выбираем несколько интересных слоев
        selected_layers = [0, len(layers)//4, len(layers)//2, 3*len(layers)//4, len(layers)-1]
        selected_layers = [l for l in selected_layers if l < len(layers)]
        
        for i, layer_idx in enumerate(selected_layers[:6]):
            if i >= len(axes):
                break
                
            similarity_matrix = np.array(layer_results[str(layers[layer_idx])]['similarity_matrix'])
            
            im = axes[i].imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
            axes[i].set_title(f'Layer {layers[layer_idx]} (Pairs: {num_pairs[layer_idx]})')
            axes[i].set_xlabel('Head Index')
            axes[i].set_ylabel('Head Index')
            
            # Добавляем colorbar
            plt.colorbar(im, ax=axes[i])
        
        # Скрываем неиспользуемые subplot'ы
        for i in range(len(selected_layers), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Head Similarity Matrices (Threshold: {threshold})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'similarity_matrices_threshold_{threshold}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def print_summary(results):
    """Вывести сводку результатов"""
    print("=" * 60)
    print("SUMMARY OF HEAD MERGING EXPERIMENTS")
    print("=" * 60)
    
    thresholds = sorted(results.keys())
    
    original_perplexity = results[thresholds[0]]['original_metrics']['perplexity']
    original_accuracy = results[thresholds[0]]['original_metrics']['accuracy']
    
    print(f"Original Model Performance:")
    print(f"  Perplexity: {original_perplexity:.4f}")
    print(f"  Accuracy:   {original_accuracy:.4f}")
    print()
    
    print("Results by Threshold:")
    print("-" * 60)
    print(f"{'Threshold':<10} {'Pairs':<8} {'Perplexity':<12} {'Accuracy':<10} {'Δ Perp':<10} {'Δ Acc':<10}")
    print("-" * 60)
    
    for threshold in thresholds:
        data = results[threshold]
        merged_perp = data['merged_metrics']['perplexity']
        merged_acc = data['merged_metrics']['accuracy']
        total_pairs = data['total_merged_pairs']
        
        delta_perp = merged_perp - original_perplexity
        delta_acc = merged_acc - original_accuracy
        
        print(f"{threshold:<10.3f} {total_pairs:<8} {merged_perp:<12.4f} {merged_acc:<10.4f} "
              f"{delta_perp:<10.4f} {delta_acc:<10.4f}")
    
    print("-" * 60)
    
    # Найдем лучший порог
    best_threshold = min(thresholds, key=lambda t: results[t]['merged_metrics']['perplexity'])
    best_data = results[best_threshold]
    
    print(f"\nBest Threshold: {best_threshold}")
    print(f"  Merged Pairs: {best_data['total_merged_pairs']}")
    print(f"  Perplexity: {original_perplexity:.4f} -> {best_data['merged_metrics']['perplexity']:.4f}")
    print(f"  Accuracy: {original_accuracy:.4f} -> {best_data['merged_metrics']['accuracy']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze head merging results")
    parser.add_argument("--results_dir", default="results/head_merging", 
                       help="Directory containing results")
    parser.add_argument("--output_dir", default="results/head_merging/analysis", 
                       help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем результаты
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Found results for {len(results)} thresholds: {sorted(results.keys())}")
    
    # Создаем анализ
    create_threshold_comparison(results, args.output_dir)
    create_layer_analysis(results, args.output_dir)
    
    # Выводим сводку
    print_summary(results)
    
    print(f"\nAnalysis saved to {args.output_dir}")

if __name__ == "__main__":
    main()
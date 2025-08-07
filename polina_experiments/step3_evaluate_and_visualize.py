#!/usr/bin/env python3
"""
Шаг 3: Оценка и визуализация результатов объединения голов
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import sys
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'igor_exps'))

from pruninghealing.dataset import DatasetLoader
from pruninghealing.utils import calculate_perplexity

class ModelEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        
    def load_model(self, model_path):
        """Загрузить модель и токенизатор"""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    
    def load_c4_dataset(self, tokenizer):
        """Загрузить датасет C4 (как в igor_exps/iterative_pruning.py)"""
        dataset_loader = DatasetLoader(tokenizer)
        dataset_loader.load_c4(train_size=500, eval_size=100)
        return dataset_loader
    
    def compute_accuracy(self, model, tokenizer, dataset_loader, max_samples=50):
        """Вычислить accuracy на тестовом наборе"""
        model.eval()
        correct = 0
        total = 0
        
        # Используем eval dataset
        eval_data = dataset_loader.eval_dataset.select(range(min(max_samples, len(dataset_loader.eval_dataset))))
        
        with torch.no_grad():
            for example in tqdm(eval_data, desc="Computing accuracy", leave=False):
                input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(self.device)
                labels = torch.tensor(example['labels']).unsqueeze(0).to(self.device)
                
                if input_ids.size(1) < 2:
                    continue
                    
                # Предсказываем следующий токен
                outputs = model(input_ids[:, :-1])
                predictions = outputs.logits.argmax(dim=-1)
                targets = labels[:, 1:]
                
                # Считаем точность
                mask = targets != -100
                correct += (predictions == targets)[mask].sum().item()
                total += mask.sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_model(self, model, tokenizer, dataset_loader):
        """Полная оценка модели"""
        print("Computing perplexity...")
        perplexity = calculate_perplexity(
            model, tokenizer, 
            dataset=dataset_loader.eval_dataset,
            max_samples=50
        )
        
        print("Computing accuracy...")
        accuracy = self.compute_accuracy(model, tokenizer, dataset_loader, max_samples=50)
        
        return {
            'perplexity': perplexity,
            'accuracy': accuracy
        }

def create_comparison_plots(original_metrics, merged_metrics, merge_log, output_dir):
    """Создать графики сравнения метрик"""
    
    # График 1: Сравнение метрик
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    metrics = ['Original', 'After Merging']
    perplexities = [original_metrics['perplexity'], merged_metrics['perplexity']]
    accuracies = [original_metrics['accuracy'], merged_metrics['accuracy']]
    
    # Perplexity
    bars1 = ax1.bar(metrics, perplexities, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars1, perplexities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Accuracy
    bars2 = ax2.bar(metrics, accuracies, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # График 2: Количество объединенных пар по слоям
    if merge_log['merged_layers']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        layers = sorted([int(k) for k in merge_log['merged_layers'].keys()])
        num_pairs = [merge_log['merged_layers'][str(layer)]['num_pairs_merged'] for layer in layers]
        
        bars = ax.bar(layers, num_pairs, alpha=0.7, color='skyblue', edgecolor='navy')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Number of Merged Pairs')
        ax.set_title(f'Merged Head Pairs per Layer (threshold={merge_log["threshold"]})')
        ax.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, num_pairs):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'merged_pairs_per_layer.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # График 3: Изменение метрик
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics_names = ['Perplexity', 'Accuracy']
    original_values = [original_metrics['perplexity'], original_metrics['accuracy']]
    merged_values = [merged_metrics['perplexity'], merged_metrics['accuracy']]
    changes = [(merged - orig) / orig * 100 for orig, merged in zip(original_values, merged_values)]
    
    colors = ['red' if change > 0 else 'green' for change in changes]
    bars = ax.bar(metrics_names, changes, color=colors, alpha=0.7)
    
    ax.set_ylabel('Change (%)')
    ax.set_title('Relative Change in Metrics After Head Merging')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, change in zip(bars, changes):
        ax.text(bar.get_x() + bar.get_width()/2, 
               bar.get_height() + (0.1 if change > 0 else -0.3),
               f'{change:+.2f}%', ha='center', va='bottom' if change > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_change.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize head merging results")
    parser.add_argument("--original_model_path", default="src/checkpoints/llama3.1-8b",
                       help="Path to original model")
    parser.add_argument("--merged_model_path", required=True,
                       help="Path to merged model")
    parser.add_argument("--output_dir", default="polina_experiments/results",
                       help="Output directory for results")
    parser.add_argument("--device", default="cuda:3", help="Device to use")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = ModelEvaluator(args.device)
    
    # Загружаем модели
    print("Loading original model...")
    original_model, tokenizer = evaluator.load_model(args.original_model_path)
    
    print("Loading merged model...")
    merged_model, _ = evaluator.load_model(args.merged_model_path)
    
    # Загружаем датасет C4
    print("Loading C4 dataset...")
    dataset_loader = evaluator.load_c4_dataset(tokenizer)
    
    # Оцениваем оригинальную модель
    print("\nEvaluating original model...")
    original_metrics = evaluator.evaluate_model(original_model, tokenizer, dataset_loader)
    print(f"Original - Perplexity: {original_metrics['perplexity']:.4f}, Accuracy: {original_metrics['accuracy']:.4f}")
    
    # Оцениваем объединенную модель
    print("\nEvaluating merged model...")
    merged_metrics = evaluator.evaluate_model(merged_model, tokenizer, dataset_loader)
    print(f"Merged - Perplexity: {merged_metrics['perplexity']:.4f}, Accuracy: {merged_metrics['accuracy']:.4f}")
    
    # Загружаем лог объединения
    merge_log_file = os.path.join(args.merged_model_path, "merge_log.json")
    if os.path.exists(merge_log_file):
        with open(merge_log_file, 'r') as f:
            merge_log = json.load(f)
    else:
        merge_log = {'merged_layers': {}, 'threshold': 'unknown', 'total_merged_pairs': 0}
    
    # Создаем результаты
    results = {
        'original_model_path': args.original_model_path,
        'merged_model_path': args.merged_model_path,
        'original_metrics': original_metrics,
        'merged_metrics': merged_metrics,
        'merge_info': merge_log,
        'metrics_change': {
            'perplexity_change': merged_metrics['perplexity'] - original_metrics['perplexity'],
            'accuracy_change': merged_metrics['accuracy'] - original_metrics['accuracy'],
            'perplexity_change_percent': (merged_metrics['perplexity'] - original_metrics['perplexity']) / original_metrics['perplexity'] * 100,
            'accuracy_change_percent': (merged_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy'] * 100
        }
    }
    
    # Сохраняем результаты
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Создаем графики
    print("\nCreating visualization plots...")
    create_comparison_plots(original_metrics, merged_metrics, merge_log, args.output_dir)
    
    # Выводим итоговую статистику
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total merged pairs: {merge_log.get('total_merged_pairs', 0)}")
    print(f"Threshold used: {merge_log.get('threshold', 'unknown')}")
    print(f"\nMetrics comparison:")
    print(f"  Perplexity: {original_metrics['perplexity']:.4f} -> {merged_metrics['perplexity']:.4f} ({results['metrics_change']['perplexity_change_percent']:+.2f}%)")
    print(f"  Accuracy:   {original_metrics['accuracy']:.4f} -> {merged_metrics['accuracy']:.4f} ({results['metrics_change']['accuracy_change_percent']:+.2f}%)")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Evaluation data: {results_file}")
    print(f"Plots: metrics_comparison.png, merged_pairs_per_layer.png, metrics_change.png")

if __name__ == "__main__":
    main()
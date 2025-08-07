#!/usr/bin/env python3
"""
Шаг 3: Визуализация результатов объединения голов
"""

import torch
import matplotlib.pyplot as plt
import json
import argparse
import os
import sys
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pruninghealing.dataset import DatasetLoader
from pruninghealing.utils import calculate_perplexity

class ResultsVisualizer:
    def __init__(self, original_model_path, merged_model_path, device="cuda"):
        self.device = device
        self.original_model, self.tokenizer = self.load_model(original_model_path)
        self.merged_model, _ = self.load_model(merged_model_path)
        
    def load_model(self, model_path):
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
    
    def compute_accuracy(self, model, dataset_loader, max_samples=50):
        """Вычислить accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        eval_data = dataset_loader.eval_dataset.select(range(min(max_samples, len(dataset_loader.eval_dataset))))
        
        with torch.no_grad():
            for example in tqdm(eval_data, desc="Computing accuracy"):
                input_ids = torch.tensor(example['input_ids']).unsqueeze(0)
                labels = torch.tensor(example['labels']).unsqueeze(0)
                
                if input_ids.size(1) < 2:
                    continue
                    
                outputs = model(input_ids[:, :-1])
                predictions = outputs.logits.argmax(dim=-1)
                targets = labels[:, 1:]
                
                mask = targets != -100
                correct += (predictions == targets)[mask].sum().item()
                total += mask.sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_models(self, dataset_loader):
        """Оценить обе модели"""
        print("Evaluating original model...")
        original_perplexity = calculate_perplexity(
            self.original_model, self.tokenizer, 
            dataset=dataset_loader.eval_dataset, max_samples=20
        )
        original_accuracy = self.compute_accuracy(self.original_model, dataset_loader, max_samples=50)
        
        print("Evaluating merged model...")
        merged_perplexity = calculate_perplexity(
            self.merged_model, self.tokenizer, 
            dataset=dataset_loader.eval_dataset, max_samples=20
        )
        merged_accuracy = self.compute_accuracy(self.merged_model, dataset_loader, max_samples=50)
        
        return {
            'original': {'perplexity': original_perplexity, 'accuracy': original_accuracy},
            'merged': {'perplexity': merged_perplexity, 'accuracy': merged_accuracy}
        }
    
    def create_comparison_plots(self, metrics, merge_log, output_dir):
        """Создать графики сравнения"""
        # График 1: Сравнение метрик
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        models = ['Original', 'Merged']
        perplexities = [metrics['original']['perplexity'], metrics['merged']['perplexity']]
        accuracies = [metrics['original']['accuracy'], metrics['merged']['accuracy']]
        
        # Perplexity
        bars1 = ax1.bar(models, perplexities, color=['blue', 'red'], alpha=0.7)
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Perplexity Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, val in zip(bars1, perplexities):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Accuracy
        bars2 = ax2.bar(models, accuracies, color=['blue', 'red'], alpha=0.7)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, val in zip(bars2, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # График 2: Количество объединенных пар по слоям
        if 'merged_layers' in merge_log:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            layers = sorted([int(k) for k in merge_log['merged_layers'].keys()])
            num_pairs = [merge_log['merged_layers'][str(layer)]['num_pairs_merged'] for layer in layers]
            
            bars = ax.bar(layers, num_pairs, alpha=0.7, color='skyblue', edgecolor='navy')
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Number of Merged Pairs')
            ax.set_title(f'Merged Head Pairs per Layer (threshold={merge_log.get("threshold", "N/A")})')
            ax.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar, val in zip(bars, num_pairs):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           str(val), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'merged_pairs_per_layer.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # График 3: Изменение метрик
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics_names = ['Perplexity', 'Accuracy']
        original_vals = [metrics['original']['perplexity'], metrics['original']['accuracy']]
        merged_vals = [metrics['merged']['perplexity'], metrics['merged']['accuracy']]
        changes = [(merged_vals[i] - original_vals[i]) / original_vals[i] * 100 for i in range(2)]
        
        colors = ['red' if change < 0 else 'green' for change in changes]
        bars = ax.bar(metrics_names, changes, color=colors, alpha=0.7)
        
        ax.set_ylabel('Change (%)')
        ax.set_title('Metrics Change After Head Merging')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, val in zip(bars, changes):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + (0.1 if val > 0 else -0.3),
                   f'{val:+.2f}%', ha='center', va='bottom' if val > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_change.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize head merging results")
    parser.add_argument("--original_model_path", default="src/checkpoints/llama3.1-8b", 
                       help="Path to original model")
    parser.add_argument("--merged_model_path", required=True,
                       help="Path to merged model")
    parser.add_argument("--merge_log_file", required=True,
                       help="JSON file with merge log")
    parser.add_argument("--output_dir", default="polina_experiments/results/visualization",
                       help="Output directory for plots")
    parser.add_argument("--device", default="cuda:3", help="Device to use")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем лог объединения
    with open(args.merge_log_file, 'r') as f:
        merge_log = json.load(f)
    
    # Инициализируем визуализатор
    visualizer = ResultsVisualizer(args.original_model_path, args.merged_model_path, args.device)
    
    # Загружаем датасет C4
    print("Loading C4 dataset...")
    dataset_loader = DatasetLoader(visualizer.tokenizer)
    dataset_loader.load_c4(train_size=100, eval_size=100)
    
    # Оцениваем модели
    print("Evaluating models...")
    metrics = visualizer.evaluate_models(dataset_loader)
    
    print(f"Original model - Perplexity: {metrics['original']['perplexity']:.4f}, Accuracy: {metrics['original']['accuracy']:.4f}")
    print(f"Merged model - Perplexity: {metrics['merged']['perplexity']:.4f}, Accuracy: {metrics['merged']['accuracy']:.4f}")
    
    # Создаем графики
    print("Creating visualization plots...")
    visualizer.create_comparison_plots(metrics, merge_log, args.output_dir)
    
    # Сохраняем результаты
    results = {
        'original_model_path': args.original_model_path,
        'merged_model_path': args.merged_model_path,
        'merge_log_file': args.merge_log_file,
        'metrics': metrics,
        'total_merged_pairs': merge_log.get('total_merged_pairs', 0),
        'threshold': merge_log.get('threshold', 'N/A')
    }
    
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nVisualization completed!")
    print(f"Plots saved to: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    
    # Выводим краткую сводку
    perp_change = (metrics['merged']['perplexity'] - metrics['original']['perplexity']) / metrics['original']['perplexity'] * 100
    acc_change = (metrics['merged']['accuracy'] - metrics['original']['accuracy']) / metrics['original']['accuracy'] * 100
    
    print(f"\nSummary:")
    print(f"Total merged pairs: {merge_log.get('total_merged_pairs', 0)}")
    print(f"Perplexity change: {perp_change:+.2f}%")
    print(f"Accuracy change: {acc_change:+.2f}%")

if __name__ == "__main__":
    main()
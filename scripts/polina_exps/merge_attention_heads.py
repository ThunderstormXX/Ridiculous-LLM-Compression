#!/usr/bin/env python3
"""
Эксперимент: объединение attention-голов в модели Llama3.1-8B на основе их схожести.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaForCausalLM, AutoTokenizer
import argparse
import os
import json
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pruninghealing.dataset import DatasetLoader
from pruninghealing.utils import calculate_perplexity

class AttentionHeadMerger:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model, self.tokenizer = self.load_model(model_path)
        self.original_weights = {}
        
    def load_model(self, model_path):
        """Загрузить модель и токенизатор"""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    
    def extract_attention_weights(self, layer_idx):
        """Извлечь веса Q, K, V, O для всех голов слоя"""
        layer = self.model.model.layers[layer_idx]
        
        # Получаем параметры архитектуры
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads
        
        # Извлекаем веса
        q_weight = layer.self_attn.q_proj.weight.data  # [hidden_size, hidden_size]
        k_weight = layer.self_attn.k_proj.weight.data  # [num_kv_heads * head_dim, hidden_size]
        v_weight = layer.self_attn.v_proj.weight.data  # [num_kv_heads * head_dim, hidden_size]
        o_weight = layer.self_attn.o_proj.weight.data  # [hidden_size, hidden_size]
        
        # Разделяем веса по головам
        q_heads = q_weight.view(num_heads, head_dim, -1)  # [num_heads, head_dim, hidden_size]
        o_heads = o_weight.view(-1, num_heads, head_dim).transpose(0, 1)  # [num_heads, hidden_size, head_dim]
        
        return {
            'q_heads': q_heads,
            'k_weight': k_weight,
            'v_weight': v_weight,
            'o_heads': o_heads,
            'num_heads': num_heads,
            'head_dim': head_dim
        }
    
    def compute_head_similarity(self, layer_idx, metric='cosine'):
        """Вычислить схожесть между головами слоя"""
        weights = self.extract_attention_weights(layer_idx)
        q_heads = weights['q_heads']
        num_heads = weights['num_heads']
        
        # Преобразуем веса голов в векторы для сравнения
        head_vectors = q_heads.view(num_heads, -1).cpu().numpy()
        
        if metric == 'cosine':
            similarity_matrix = cosine_similarity(head_vectors)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        return similarity_matrix
    
    def find_similar_pairs(self, similarity_matrix, threshold=0.99):
        """Найти пары голов с схожестью выше порога"""
        num_heads = similarity_matrix.shape[0]
        pairs = []
        
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                if similarity_matrix[i, j] >= threshold:
                    pairs.append((i, j, similarity_matrix[i, j]))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)
    
    def backup_weights(self, layer_idx):
        """Сохранить оригинальные веса слоя"""
        layer = self.model.model.layers[layer_idx]
        self.original_weights[layer_idx] = {
            'q_proj': layer.self_attn.q_proj.weight.data.clone(),
            'k_proj': layer.self_attn.k_proj.weight.data.clone(),
            'v_proj': layer.self_attn.v_proj.weight.data.clone(),
            'o_proj': layer.self_attn.o_proj.weight.data.clone()
        }
    
    def merge_heads(self, layer_idx, pairs):
        """Объединить похожие головы путем усреднения весов"""
        if not pairs:
            return
            
        self.backup_weights(layer_idx)
        layer = self.model.model.layers[layer_idx]
        
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads
        
        # Получаем текущие веса
        q_weight = layer.self_attn.q_proj.weight.data
        o_weight = layer.self_attn.o_proj.weight.data
        
        # Разделяем по головам
        q_heads = q_weight.view(num_heads, head_dim, -1)
        o_heads = o_weight.view(-1, num_heads, head_dim).transpose(0, 1)
        
        # Объединяем пары
        for head1, head2, similarity in pairs:
            # Усредняем веса Q
            avg_q = (q_heads[head1] + q_heads[head2]) / 2
            q_heads[head1] = avg_q.clone()
            q_heads[head2] = avg_q.clone()
            
            # Усредняем веса O
            avg_o = (o_heads[head1] + o_heads[head2]) / 2
            o_heads[head1] = avg_o.clone()
            o_heads[head2] = avg_o.clone()
        
        # Записываем обратно
        layer.self_attn.q_proj.weight.data = q_heads.view(num_heads * head_dim, -1)
        layer.self_attn.o_proj.weight.data = o_heads.transpose(0, 1).contiguous().view(-1, num_heads * head_dim)
    
    def restore_weights(self, layer_idx):
        """Восстановить оригинальные веса слоя"""
        if layer_idx not in self.original_weights:
            return
            
        layer = self.model.model.layers[layer_idx]
        weights = self.original_weights[layer_idx]
        
        layer.self_attn.q_proj.weight.data.copy_(weights['q_proj'])
        layer.self_attn.k_proj.weight.data.copy_(weights['k_proj'])
        layer.self_attn.v_proj.weight.data.copy_(weights['v_proj'])
        layer.self_attn.o_proj.weight.data.copy_(weights['o_proj'])
    
    def evaluate_model(self, dataset_loader, max_samples=100):
        """Оценить качество модели"""
        # Вычисляем perplexity
        perplexity = calculate_perplexity(
            self.model, self.tokenizer, 
            max_samples=max_samples
        )
        
        # Простая оценка accuracy на небольшом наборе
        accuracy = self.compute_accuracy(dataset_loader, max_samples=50)
        
        return {
            'perplexity': perplexity,
            'accuracy': accuracy
        }
    
    def compute_accuracy(self, dataset_loader, max_samples=50):
        """Вычислить accuracy на тестовом наборе"""
        self.model.eval()
        correct = 0
        total = 0
        
        # Используем eval dataset
        eval_data = dataset_loader.eval_dataset.select(range(min(max_samples, len(dataset_loader.eval_dataset))))
        
        with torch.no_grad():
            for example in tqdm(eval_data, desc="Computing accuracy"):
                input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(self.device)
                labels = torch.tensor(example['labels']).unsqueeze(0).to(self.device)
                
                if input_ids.size(1) < 2:
                    continue
                    
                # Предсказываем следующий токен
                outputs = self.model(input_ids[:, :-1])
                predictions = outputs.logits.argmax(dim=-1)
                targets = labels[:, 1:]
                
                # Считаем точность
                mask = targets != -100
                correct += (predictions == targets)[mask].sum().item()
                total += mask.sum().item()
        
        return correct / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Merge attention heads based on similarity")
    parser.add_argument("--model_path", default="src/checkpoints/llama3.1-8b", 
                       help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.99, 
                       help="Cosine similarity threshold for merging")
    parser.add_argument("--layers", nargs="+", type=int, 
                       help="Specific layers to analyze (default: all)")
    parser.add_argument("--output_dir", default="results/head_merging", 
                       help="Output directory for results")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Инициализируем merger
    merger = AttentionHeadMerger(args.model_path, args.device)
    
    # Загружаем датасет
    dataset_loader = DatasetLoader(merger.tokenizer)
    dataset_loader.load_wikitext(max_length=512, train_size=1000, eval_size=200)
    
    # Определяем слои для анализа
    num_layers = len(merger.model.model.layers)
    if args.layers:
        layers_to_analyze = args.layers
    else:
        layers_to_analyze = list(range(num_layers))
    
    print(f"Analyzing {len(layers_to_analyze)} layers with threshold {args.threshold}")
    
    # Оценка до объединения
    print("Evaluating original model...")
    original_metrics = merger.evaluate_model(dataset_loader)
    print(f"Original - Perplexity: {original_metrics['perplexity']:.4f}, Accuracy: {original_metrics['accuracy']:.4f}")
    
    # Анализ и объединение голов
    results = {
        'threshold': args.threshold,
        'original_metrics': original_metrics,
        'layer_results': {},
        'merged_metrics': {}
    }
    
    total_merged_pairs = 0
    
    for layer_idx in tqdm(layers_to_analyze, desc="Processing layers"):
        # Вычисляем схожесть
        similarity_matrix = merger.compute_head_similarity(layer_idx)
        
        # Находим похожие пары
        pairs = merger.find_similar_pairs(similarity_matrix, args.threshold)
        
        if pairs:
            print(f"Layer {layer_idx}: Found {len(pairs)} similar pairs")
            total_merged_pairs += len(pairs)
            
            # Объединяем головы
            merger.merge_heads(layer_idx, pairs)
            
            results['layer_results'][layer_idx] = {
                'num_pairs': len(pairs),
                'pairs': [(int(p[0]), int(p[1]), float(p[2])) for p in pairs],
                'similarity_matrix': similarity_matrix.tolist()
            }
        else:
            print(f"Layer {layer_idx}: No similar pairs found")
            results['layer_results'][layer_idx] = {
                'num_pairs': 0,
                'pairs': [],
                'similarity_matrix': similarity_matrix.tolist()
            }
    
    # Оценка после объединения
    print(f"\nEvaluating model after merging {total_merged_pairs} pairs...")
    merged_metrics = merger.evaluate_model(dataset_loader)
    print(f"Merged - Perplexity: {merged_metrics['perplexity']:.4f}, Accuracy: {merged_metrics['accuracy']:.4f}")
    
    results['merged_metrics'] = merged_metrics
    results['total_merged_pairs'] = total_merged_pairs
    
    # Сохраняем результаты
    results_file = os.path.join(args.output_dir, f"results_threshold_{args.threshold}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Создаем график сравнения
    create_comparison_plot(results, args.output_dir, args.threshold)
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Perplexity change: {original_metrics['perplexity']:.4f} -> {merged_metrics['perplexity']:.4f}")
    print(f"Accuracy change: {original_metrics['accuracy']:.4f} -> {merged_metrics['accuracy']:.4f}")

def create_comparison_plot(results, output_dir, threshold):
    """Создать график сравнения метрик"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # График perplexity по слоям
    layers = sorted([int(k) for k in results['layer_results'].keys()])
    original_perp = [results['original_metrics']['perplexity']] * len(layers)
    merged_perp = [results['merged_metrics']['perplexity']] * len(layers)
    
    ax1.plot(layers, original_perp, 'b-', label='Original', linewidth=2)
    ax1.plot(layers, merged_perp, 'r-', label='After merging', linewidth=2)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График accuracy по слоям
    original_acc = [results['original_metrics']['accuracy']] * len(layers)
    merged_acc = [results['merged_metrics']['accuracy']] * len(layers)
    
    ax2.plot(layers, original_acc, 'b-', label='Original', linewidth=2)
    ax2.plot(layers, merged_acc, 'r-', label='After merging', linewidth=2)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_threshold_{threshold}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # График количества объединенных пар по слоям
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    num_pairs = [results['layer_results'][str(layer)]['num_pairs'] for layer in layers]
    
    ax.bar(layers, num_pairs, alpha=0.7, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Merged Pairs')
    ax.set_title(f'Merged Head Pairs per Layer (threshold={threshold})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'merged_pairs_threshold_{threshold}.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
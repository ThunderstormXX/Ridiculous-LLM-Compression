#!/usr/bin/env python3
"""
Унифицированный эксперимент по объединению attention heads
Разделен на 3 этапа: поиск похожих голов, объединение, визуализация
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import sys
from tqdm import tqdm
from transformers import LlamaForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pruninghealing.dataset import DatasetLoader
from src.pruninghealing.utils import calculate_perplexity

class UnifiedHeadMergingExperiment:
    def __init__(self, model_path, device="cpu"):
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
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    
    def compute_head_similarity_simple(self, layer_idx):
        """Упрощенное вычисление сходства на основе весов Q и K"""
        layer = self.model.model.layers[layer_idx]
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads
        
        # Получаем веса Q и K
        q_weight = layer.self_attn.q_proj.weight.data
        k_weight = layer.self_attn.k_proj.weight.data
        
        # Разделяем по головам
        q_heads = q_weight.view(num_heads, head_dim, -1)
        num_kv_heads = k_weight.shape[0] // head_dim
        k_heads = k_weight.view(num_kv_heads, head_dim, -1)
        
        # Вычисляем QK^T для каждой головы и преобразуем в векторы
        qk_vectors = []
        for head_idx in range(num_heads):
            k_idx = head_idx // (num_heads // num_kv_heads) if num_kv_heads < num_heads else head_idx
            k_idx = min(k_idx, num_kv_heads - 1)
            
            # QK^T произведение
            qk_product = torch.matmul(q_heads[head_idx], k_heads[k_idx].T)  # [head_dim, head_dim]
            qk_vectors.append(qk_product.flatten().cpu().numpy())
        
        qk_vectors = np.array(qk_vectors)
        
        # Вычисляем косинусное сходство
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(qk_vectors)
        
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
    
    def step1_find_similar_heads(self, threshold=0.99):
        """Шаг 1: Поиск похожих голов"""
        print(f"=== STEP 1: Finding similar heads (threshold={threshold}) ===")
        
        num_layers = len(self.model.model.layers)
        results = {
            'threshold': threshold,
            'layer_analysis': {}
        }
        
        total_similar_pairs = 0
        
        for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
            # Вычисляем сходство
            similarity_matrix = self.compute_head_similarity_simple(layer_idx)
            
            # Находим похожие пары
            pairs = self.find_similar_pairs(similarity_matrix, threshold)
            
            results['layer_analysis'][layer_idx] = {
                'num_heads': similarity_matrix.shape[0],
                'similar_pairs': [(int(p[0]), int(p[1]), float(p[2])) for p in pairs],
                'num_similar_pairs': len(pairs),
                'similarity_matrix': similarity_matrix.tolist()
            }
            
            if pairs:
                print(f"Layer {layer_idx}: Found {len(pairs)} similar pairs:")
                for head1, head2, sim in pairs:
                    print(f"  Head {head1} <-> Head {head2}: {sim:.4f}")
                total_similar_pairs += len(pairs)
            else:
                print(f"Layer {layer_idx}: No similar pairs found")
        
        print(f"Total similar pairs found: {total_similar_pairs}")
        return results
    
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
        v_weight = layer.self_attn.v_proj.weight.data
        o_weight = layer.self_attn.o_proj.weight.data
        
        # Разделяем по головам
        q_heads = q_weight.view(num_heads, head_dim, -1)
        v_heads = v_weight.view(num_heads, head_dim, -1)
        o_heads = o_weight.view(-1, num_heads, head_dim).transpose(0, 1)
        
        # Объединяем пары путем усреднения
        for head1, head2, similarity in pairs:
            # Усредняем веса Q
            avg_q = (q_heads[head1] + q_heads[head2]) / 2
            q_heads[head1] = avg_q.clone()
            q_heads[head2] = avg_q.clone()
            
            # Усредняем веса V
            avg_v = (v_heads[head1] + v_heads[head2]) / 2
            v_heads[head1] = avg_v.clone()
            v_heads[head2] = avg_v.clone()
            
            # Усредняем веса O
            avg_o = (o_heads[head1] + o_heads[head2]) / 2
            o_heads[head1] = avg_o.clone()
            o_heads[head2] = avg_o.clone()
        
        # Записываем обратно
        layer.self_attn.q_proj.weight.data = q_heads.view(num_heads * head_dim, -1)
        layer.self_attn.v_proj.weight.data = v_heads.view(num_heads * head_dim, -1)
        layer.self_attn.o_proj.weight.data = o_heads.transpose(0, 1).contiguous().view(-1, num_heads * head_dim)
    
    def compute_accuracy(self, dataset_loader, max_samples=20):
        """Вычислить accuracy на тестовом наборе"""
        self.model.eval()
        correct = 0
        total = 0
        
        eval_data = dataset_loader.eval_dataset.select(range(min(max_samples, len(dataset_loader.eval_dataset))))
        
        with torch.no_grad():
            for example in eval_data:
                input_ids = torch.tensor(example['input_ids']).unsqueeze(0)
                labels = torch.tensor(example['labels']).unsqueeze(0)
                
                if input_ids.size(1) < 2:
                    continue
                    
                outputs = self.model(input_ids[:, :-1])
                predictions = outputs.logits.argmax(dim=-1)
                targets = labels[:, 1:]
                
                mask = targets != -100
                correct += (predictions == targets)[mask].sum().item()
                total += mask.sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def step2_merge_heads(self, similarity_results, dataset_loader):
        """Шаг 2: Объединение похожих голов"""
        print("=== STEP 2: Merging similar heads ===")
        
        # Оценка до объединения
        print("Evaluating original model...")
        original_perplexity = calculate_perplexity(
            self.model, self.tokenizer, 
            dataset=dataset_loader.eval_dataset, max_samples=10
        )
        original_accuracy = self.compute_accuracy(dataset_loader, max_samples=20)
        
        print(f"Original - Perplexity: {original_perplexity:.4f}, Accuracy: {original_accuracy:.4f}")
        
        # Объединяем головы по слоям
        total_merged_pairs = 0
        layer_merge_info = {}
        
        for layer_idx, layer_data in similarity_results['layer_analysis'].items():
            pairs = [(p[0], p[1], p[2]) for p in layer_data['similar_pairs']]
            
            if pairs:
                print(f"Merging {len(pairs)} pairs in layer {layer_idx}")
                self.merge_heads(int(layer_idx), pairs)
                total_merged_pairs += len(pairs)
                layer_merge_info[int(layer_idx)] = {
                    'merged_pairs': pairs,
                    'num_merged': len(pairs)
                }
        
        print(f"Total merged pairs: {total_merged_pairs}")
        
        # Оценка после объединения
        print("Evaluating merged model...")
        merged_perplexity = calculate_perplexity(
            self.model, self.tokenizer,
            dataset=dataset_loader.eval_dataset, max_samples=10
        )
        merged_accuracy = self.compute_accuracy(dataset_loader, max_samples=20)
        
        print(f"Merged - Perplexity: {merged_perplexity:.4f}, Accuracy: {merged_accuracy:.4f}")
        
        return {
            'original_metrics': {
                'perplexity': original_perplexity,
                'accuracy': original_accuracy
            },
            'merged_metrics': {
                'perplexity': merged_perplexity,
                'accuracy': merged_accuracy
            },
            'total_merged_pairs': total_merged_pairs,
            'layer_merge_info': layer_merge_info,
            'threshold': similarity_results['threshold']
        }
    
    def step3_visualize_results(self, merge_results, similarity_results, output_dir):
        """Шаг 3: Визуализация результатов"""
        print("=== STEP 3: Creating visualizations ===")
        
        # График 1: Сравнение метрик
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Perplexity
        metrics = ['Original', 'After Merging']
        perplexity_values = [
            merge_results['original_metrics']['perplexity'],
            merge_results['merged_metrics']['perplexity']
        ]
        
        bars1 = ax1.bar(metrics, perplexity_values, color=['blue', 'red'], alpha=0.7)
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Perplexity Comparison')
        ax1.grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, perplexity_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Accuracy
        accuracy_values = [
            merge_results['original_metrics']['accuracy'],
            merge_results['merged_metrics']['accuracy']
        ]
        
        bars2 = ax2.bar(metrics, accuracy_values, color=['blue', 'red'], alpha=0.7)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Comparison')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, accuracy_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # График 2: Количество объединенных пар по слоям
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        layers = []
        merged_pairs = []
        
        for layer_idx_str, layer_data in similarity_results['layer_analysis'].items():
            layers.append(int(layer_idx_str))
            merged_pairs.append(layer_data['num_similar_pairs'])
        
        bars = ax.bar(layers, merged_pairs, alpha=0.7, color='skyblue', edgecolor='navy')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Number of Similar Head Pairs')
        ax.set_title(f'Similar Head Pairs per Layer (threshold={merge_results["threshold"]})')
        ax.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, merged_pairs):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'merged_pairs_per_layer.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations created successfully!")
    
    def run_full_experiment(self, threshold=0.99, output_dir="polina_experiments/results"):
        """Запустить полный эксперимент"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Загружаем C4 датасет
        print("Loading C4 dataset...")
        dataset_loader = DatasetLoader(self.tokenizer)
        dataset_loader.load_c4(max_length=256, train_size=100, eval_size=20)
        
        # Шаг 1: Поиск похожих голов
        similarity_results = self.step1_find_similar_heads(threshold)
        
        # Сохраняем результаты шага 1
        similarity_file = os.path.join(output_dir, f"similar_heads_threshold_{threshold}.json")
        with open(similarity_file, 'w') as f:
            json.dump(similarity_results, f, indent=2)
        print(f"Step 1 results saved to {similarity_file}")
        
        # Шаг 2: Объединение голов
        merge_results = self.step2_merge_heads(similarity_results, dataset_loader)
        
        # Сохраняем результаты шага 2
        merge_file = os.path.join(output_dir, f"merge_results_threshold_{threshold}.json")
        with open(merge_file, 'w') as f:
            json.dump(merge_results, f, indent=2)
        print(f"Step 2 results saved to {merge_file}")
        
        # Шаг 3: Визуализация
        self.step3_visualize_results(merge_results, similarity_results, output_dir)
        
        # Создаем отчет
        self.create_summary_report(merge_results, similarity_results, output_dir)
        
        print(f"\n=== EXPERIMENT COMPLETED ===")
        print(f"Results saved to: {output_dir}")
        print(f"Perplexity change: {merge_results['original_metrics']['perplexity']:.4f} -> {merge_results['merged_metrics']['perplexity']:.4f}")
        print(f"Accuracy change: {merge_results['original_metrics']['accuracy']:.4f} -> {merge_results['merged_metrics']['accuracy']:.4f}")
        
        return merge_results, similarity_results
    
    def create_summary_report(self, merge_results, similarity_results, output_dir):
        """Создать текстовый отчет"""
        report_lines = [
            "=== ATTENTION HEAD MERGING EXPERIMENT REPORT ===\n",
            f"Threshold: {merge_results['threshold']}\n",
            f"Total merged pairs: {merge_results['total_merged_pairs']}\n",
            "\n=== METRICS COMPARISON ===",
            f"Original Perplexity: {merge_results['original_metrics']['perplexity']:.4f}",
            f"Merged Perplexity:   {merge_results['merged_metrics']['perplexity']:.4f}",
            f"Perplexity Change:   {merge_results['merged_metrics']['perplexity'] - merge_results['original_metrics']['perplexity']:+.4f}",
            "",
            f"Original Accuracy: {merge_results['original_metrics']['accuracy']:.4f}",
            f"Merged Accuracy:   {merge_results['merged_metrics']['accuracy']:.4f}",
            f"Accuracy Change:   {merge_results['merged_metrics']['accuracy'] - merge_results['original_metrics']['accuracy']:+.4f}",
            "\n=== LAYER-WISE ANALYSIS ===",
        ]
        
        for layer_idx_str, layer_data in similarity_results['layer_analysis'].items():
            layer_idx = int(layer_idx_str)
            num_pairs = layer_data['num_similar_pairs']
            
            if num_pairs > 0:
                report_lines.append(f"Layer {layer_idx}: {num_pairs} similar pairs")
                for head1, head2, sim in layer_data['similar_pairs']:
                    report_lines.append(f"  Head {head1} <-> Head {head2}: {sim:.4f}")
            else:
                report_lines.append(f"Layer {layer_idx}: No similar pairs")
        
        report_text = "\n".join(report_lines)
        
        with open(os.path.join(output_dir, 'experiment_report.txt'), 'w') as f:
            f.write(report_text)

def main():
    parser = argparse.ArgumentParser(description="Unified attention head merging experiment")
    parser.add_argument("--model_path", default="src/checkpoints/tinyllama", 
                       help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.95, 
                       help="Cosine similarity threshold")
    parser.add_argument("--output_dir", default="polina_experiments/results", 
                       help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    print(f"Starting unified head merging experiment...")
    print(f"Model: {args.model_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output_dir}")
    
    # Запускаем эксперимент
    experiment = UnifiedHeadMergingExperiment(args.model_path, args.device)
    experiment.run_full_experiment(args.threshold, args.output_dir)

if __name__ == "__main__":
    main()
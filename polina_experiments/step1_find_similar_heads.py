#!/usr/bin/env python3
"""
Шаг 1: Поиск похожих attention-голов по косинусному сходству
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaForCausalLM, AutoTokenizer
import argparse
import os
import json
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class SimilarityAnalyzer:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model, self.tokenizer = self.load_model(model_path)
        
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
    
    def compute_attention_matrices(self, layer_idx, seq_len=64):
        """Вычислить матрицы внимания для всех голов слоя"""
        layer = self.model.model.layers[layer_idx]
        
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads
        
        # Создаем случайные входные векторы h_i
        h = torch.randn(seq_len, self.model.config.hidden_size)
        
        # Получаем веса Q, K
        q_weight = layer.self_attn.q_proj.weight.data
        k_weight = layer.self_attn.k_proj.weight.data
        
        # Разделяем по головам
        q_heads = q_weight.view(num_heads, head_dim, -1)
        num_kv_heads = k_weight.shape[0] // head_dim
        k_heads = k_weight.view(num_kv_heads, head_dim, -1)
        
        attention_matrices = []
        
        for head_idx in range(num_heads):
            # Определяем соответствующую K-голову
            k_idx = head_idx // (num_heads // num_kv_heads) if num_kv_heads < num_heads else head_idx
            k_idx = min(k_idx, num_kv_heads - 1)
            
            # Вычисляем Q_i и K_j для данной головы
            Q_i = torch.matmul(h, q_heads[head_idx].T)  # [seq_len, head_dim]
            K_j = torch.matmul(h, k_heads[k_idx].T)     # [seq_len, head_dim]
            
            # Вычисляем матрицу внимания: a_ij = ReLU(h_i · Q_i · K_j · h_j)
            # Упрощенная версия: a_ij = ReLU(Q_i @ K_j^T)
            attention_matrix = torch.relu(torch.matmul(Q_i, K_j.T))  # [seq_len, seq_len]
            
            attention_matrices.append(attention_matrix.flatten().numpy())
        
        return np.array(attention_matrices)
    
    def compute_head_similarity(self, layer_idx):
        """Вычислить косинусное сходство между головами"""
        attention_matrices = self.compute_attention_matrices(layer_idx)
        similarity_matrix = cosine_similarity(attention_matrices)
        return similarity_matrix
    
    def find_similar_pairs(self, similarity_matrix, threshold=0.99):
        """Найти пары голов с схожестью выше порога"""
        num_heads = similarity_matrix.shape[0]
        pairs = []
        
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                if similarity_matrix[i, j] >= threshold:
                    pairs.append((i, j, float(similarity_matrix[i, j])))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)

def main():
    parser = argparse.ArgumentParser(description="Find similar attention heads")
    parser.add_argument("--model_path", default="src/checkpoints/llama3.1-8b", 
                       help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.99, 
                       help="Cosine similarity threshold")
    parser.add_argument("--output_file", default="polina_experiments/results/similarity_analysis.json",
                       help="Output file for results")
    parser.add_argument("--device", default="cuda:3", help="Device to use")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    analyzer = SimilarityAnalyzer(args.model_path, args.device)
    
    num_layers = len(analyzer.model.model.layers)
    print(f"Analyzing {num_layers} layers with threshold {args.threshold}")
    
    results = {
        'model_path': args.model_path,
        'threshold': args.threshold,
        'num_layers': num_layers,
        'layer_analysis': {}
    }
    
    total_similar_pairs = 0
    
    for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
        similarity_matrix = analyzer.compute_head_similarity(layer_idx)
        pairs = analyzer.find_similar_pairs(similarity_matrix, args.threshold)
        
        results['layer_analysis'][layer_idx] = {
            'num_heads': similarity_matrix.shape[0],
            'similar_pairs': pairs,
            'num_similar_pairs': len(pairs),
            'similarity_matrix': similarity_matrix.tolist()
        }
        
        if pairs:
            print(f"Layer {layer_idx}: Found {len(pairs)} similar pairs")
            for head1, head2, sim in pairs:
                print(f"  Heads {head1} <-> {head2}: {sim:.4f}")
            total_similar_pairs += len(pairs)
        else:
            print(f"Layer {layer_idx}: No similar pairs found")
    
    results['total_similar_pairs'] = total_similar_pairs
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis completed!")
    print(f"Total similar pairs found: {total_similar_pairs}")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
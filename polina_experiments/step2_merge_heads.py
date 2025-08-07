#!/usr/bin/env python3
"""
Шаг 2: Объединение похожих attention-голов путем усреднения весов
"""

import torch
import json
import argparse
import os
import sys
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class HeadMerger:
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
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    
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
        k_weight = layer.self_attn.k_proj.weight.data
        v_weight = layer.self_attn.v_proj.weight.data
        o_weight = layer.self_attn.o_proj.weight.data
        
        # Разделяем по головам
        q_heads = q_weight.view(num_heads, head_dim, -1)
        o_heads = o_weight.view(-1, num_heads, head_dim).transpose(0, 1)
        
        # Для K и V (могут быть grouped)
        num_kv_heads = k_weight.shape[0] // head_dim
        k_heads = k_weight.view(num_kv_heads, head_dim, -1)
        v_heads = v_weight.view(num_kv_heads, head_dim, -1)
        
        print(f"Layer {layer_idx}: Merging {len(pairs)} pairs")
        
        # Объединяем пары
        for head1, head2, similarity in pairs:
            print(f"  Merging heads {head1} <-> {head2} (similarity: {similarity:.4f})")
            
            # Усредняем веса Q
            avg_q = (q_heads[head1] + q_heads[head2]) / 2
            q_heads[head1] = avg_q.clone()
            q_heads[head2] = avg_q.clone()
            
            # Усредняем веса O
            avg_o = (o_heads[head1] + o_heads[head2]) / 2
            o_heads[head1] = avg_o.clone()
            o_heads[head2] = avg_o.clone()
            
            # Для K и V (учитываем grouped attention)
            if num_kv_heads < num_heads:
                k_idx1 = head1 // (num_heads // num_kv_heads)
                k_idx2 = head2 // (num_heads // num_kv_heads)
                k_idx1 = min(k_idx1, num_kv_heads - 1)
                k_idx2 = min(k_idx2, num_kv_heads - 1)
                
                if k_idx1 == k_idx2:  # Только если головы используют одну и ту же K/V
                    # K и V уже усреднены для этой группы
                    pass
            else:
                # Усредняем K и V
                avg_k = (k_heads[head1] + k_heads[head2]) / 2
                k_heads[head1] = avg_k.clone()
                k_heads[head2] = avg_k.clone()
                
                avg_v = (v_heads[head1] + v_heads[head2]) / 2
                v_heads[head1] = avg_v.clone()
                v_heads[head2] = avg_v.clone()
        
        # Записываем обратно
        layer.self_attn.q_proj.weight.data = q_heads.view(num_heads * head_dim, -1)
        layer.self_attn.o_proj.weight.data = o_heads.transpose(0, 1).contiguous().view(-1, num_heads * head_dim)
        
        if num_kv_heads == num_heads:
            layer.self_attn.k_proj.weight.data = k_heads.view(num_kv_heads * head_dim, -1)
            layer.self_attn.v_proj.weight.data = v_heads.view(num_kv_heads * head_dim, -1)

def main():
    parser = argparse.ArgumentParser(description="Merge similar attention heads")
    parser.add_argument("--model_path", default="src/checkpoints/llama3.1-8b", 
                       help="Path to model checkpoint")
    parser.add_argument("--similarity_file", required=True,
                       help="JSON file with similarity analysis results")
    parser.add_argument("--output_model_path", required=True,
                       help="Path to save merged model")
    parser.add_argument("--device", default="cuda:3", help="Device to use")
    
    args = parser.parse_args()
    
    # Загружаем результаты анализа схожести
    with open(args.similarity_file, 'r') as f:
        similarity_results = json.load(f)
    
    # Инициализируем merger
    merger = HeadMerger(args.model_path, args.device)
    
    print(f"Merging heads based on analysis from: {args.similarity_file}")
    print(f"Threshold used: {similarity_results['threshold']}")
    
    total_merged_pairs = 0
    merge_log = {
        'original_model': args.model_path,
        'similarity_file': args.similarity_file,
        'threshold': similarity_results['threshold'],
        'merged_layers': {}
    }
    
    # Объединяем головы по слоям
    for layer_idx_str, layer_data in tqdm(similarity_results['layer_analysis'].items(), desc="Merging heads"):
        layer_idx = int(layer_idx_str)
        pairs = [(p[0], p[1], p[2]) for p in layer_data['similar_pairs']]
        
        if pairs:
            merger.merge_heads(layer_idx, pairs)
            total_merged_pairs += len(pairs)
            
            merge_log['merged_layers'][layer_idx] = {
                'num_pairs_merged': len(pairs),
                'pairs': pairs
            }
    
    merge_log['total_merged_pairs'] = total_merged_pairs
    
    # Сохраняем объединенную модель
    print(f"\nSaving merged model to: {args.output_model_path}")
    os.makedirs(args.output_model_path, exist_ok=True)
    merger.model.save_pretrained(args.output_model_path)
    merger.tokenizer.save_pretrained(args.output_model_path)
    
    # Сохраняем лог объединения
    merge_log_file = os.path.join(args.output_model_path, "merge_log.json")
    with open(merge_log_file, 'w') as f:
        json.dump(merge_log, f, indent=2)
    
    print(f"Head merging completed!")
    print(f"Total pairs merged: {total_merged_pairs}")
    print(f"Merged model saved to: {args.output_model_path}")
    print(f"Merge log saved to: {merge_log_file}")

if __name__ == "__main__":
    main()
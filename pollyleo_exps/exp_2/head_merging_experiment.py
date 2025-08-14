#!/usr/bin/env python3
"""
Эксперимент по объединению голов внимания и измерению перплексии.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.pruninghealing import DatasetLoader
import copy

def load_model():
    """Загрузка модели"""
    model_path = "../../src/checkpoints/tinyllama"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_validation_data(tokenizer, num_samples=50):
    """Загрузка валидационных данных C4"""
    dataset_loader = DatasetLoader(tokenizer)
    dataset_loader.load_c4(train_size=500, eval_size=50)
    
    samples = []
    for i, example in enumerate(dataset_loader.eval_dataset):
        if len(samples) >= num_samples:
            break
        
        text = example["text"]
        tokens = tokenizer(text, truncation=True, max_length=512, 
                          padding=True, return_tensors="pt")
        samples.append(tokens)
    
    return samples

def compute_perplexity(model, validation_data):
    """Вычисление перплексии"""
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for sample in validation_data:
            input_ids = sample["input_ids"].to(model.device)
            attention_mask = sample["attention_mask"].to(model.device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def load_similar_pairs():
    """Загрузка похожих пар из файла"""
    similar_file = Path("similar_heads.txt")
    pairs = []
    
    with open(similar_file, 'r', encoding='utf-8') as f:
        for line in f:
            if "Слой" in line and "Голова" in line:
                parts = line.strip().split()
                layer_idx = int(parts[1].rstrip(':'))
                head_i = int(parts[3])
                head_j = int(parts[6])
                diff = float(parts[9])
                pairs.append((layer_idx, head_i, head_j, diff))
    
    return pairs

def merge_attention_heads(model, layer_idx, head_to_keep, head_to_remove):
    """Объединение голов внимания"""
    layer = model.model.layers[layer_idx]
    
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    with torch.no_grad():
        q_weight = layer.self_attn.q_proj.weight
        start_keep = head_to_keep * head_dim
        end_keep = (head_to_keep + 1) * head_dim
        start_remove = head_to_remove * head_dim
        end_remove = (head_to_remove + 1) * head_dim
        
        q_weight[start_remove:end_remove] = q_weight[start_keep:end_keep].clone()
        
        k_weight = layer.self_attn.k_proj.weight
        k_weight[start_remove:end_remove] = k_weight[start_keep:end_keep].clone()
        
        v_weight = layer.self_attn.v_proj.weight
        v_weight[start_remove:end_remove] = v_weight[start_keep:end_keep].clone()

def run_merging_experiment(model, validation_data, similar_pairs):
    """Запуск эксперимента по объединению голов"""
    results = []
    
    initial_perplexity = compute_perplexity(model, validation_data)
    results.append({
        'step': 0,
        'merged_heads_percent': 0.0,
        'perplexity': initial_perplexity
    })
    print(f"Исходная перплексия: {initial_perplexity:.2f}")
    
    total_pairs = len(similar_pairs)
    merged_count = 0
    
    for i, (layer_idx, head_i, head_j, diff) in enumerate(similar_pairs):
        merge_attention_heads(model, layer_idx, head_i, head_j)
        merged_count += 1
        
        if (i + 1) % max(1, total_pairs // 10) == 0 or i == total_pairs - 1:
            perplexity = compute_perplexity(model, validation_data)
            percent_merged = (merged_count / total_pairs) * 100
            
            results.append({
                'step': i + 1,
                'merged_heads_percent': percent_merged,
                'perplexity': perplexity
            })
            
            print(f"Шаг {i+1}: {percent_merged:.1f}% голов объединено, перплексия: {perplexity:.2f}")
    
    return results

def save_results_and_plot(results):
    """Сохранение результатов и создание графика"""
    df = pd.DataFrame(results)
    csv_path = Path("perplexity_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Результаты сохранены в {csv_path}")
    
    plt.figure(figsize=(12, 8))
    plt.plot(df['merged_heads_percent'], df['perplexity'], 'b-o', linewidth=2, markersize=6)
    plt.xlabel('% объединённых голов')
    plt.ylabel('Перплексия')
    plt.title('Зависимость перплексии от процента объединённых голов внимания')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = Path("perplexity_trend.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сохранен в {plot_path}")

def main():
    print("Загрузка модели...")
    model, tokenizer = load_model()
    
    print("Загрузка валидационных данных...")
    validation_data = load_validation_data(tokenizer)
    
    print("Загрузка похожих пар голов...")
    similar_pairs = load_similar_pairs()
    print(f"Найдено {len(similar_pairs)} похожих пар")
    
    print("Запуск эксперимента по объединению голов...")
    results = run_merging_experiment(model, validation_data, similar_pairs)
    
    print("Сохранение результатов...")
    save_results_and_plot(results)
    
    print("Эксперимент завершен!")

if __name__ == "__main__":
    main()
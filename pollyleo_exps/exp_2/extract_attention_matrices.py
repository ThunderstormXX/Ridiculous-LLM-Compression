#!/usr/bin/env python3
"""
Извлечение матриц внимания из tinyllama на C4 датасете.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.pruninghealing import DatasetLoader

def load_model():
    """Загрузка модели tinyllama"""
    model_path = "../../src/checkpoints/tinyllama"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_attentions=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_c4_samples(tokenizer, num_samples=100, max_length=128):
    """Загрузка случайных примеров из C4"""
    dataset_loader = DatasetLoader(tokenizer)
    dataset_loader.load_c4(train_size=500, eval_size=50)
    
    samples = []
    for i, example in enumerate(dataset_loader.train_dataset):
        if len(samples) >= num_samples:
            break
        
        text = example["text"]
        tokens = tokenizer(text, truncation=True, max_length=max_length, 
                          padding=True, return_tensors="pt")
        
        if tokens["input_ids"].shape[1] >= 64:
            samples.append(tokens)
    
    return samples

def extract_attention_matrices(model, samples):
    """Извлечение матриц внимания"""
    all_attentions = []
    
    with torch.no_grad():
        for i, sample in enumerate(samples):
            print(f"Обработка примера {i+1}/{len(samples)}")
            
            input_ids = sample["input_ids"].to(model.device)
            outputs = model(input_ids, output_attentions=True)
            
            batch_attentions = []
            for layer_idx, layer_attention in enumerate(outputs.attentions):
                layer_attention = layer_attention.squeeze(0).cpu()
                batch_attentions.append(layer_attention)
            
            all_attentions.append(batch_attentions)
    
    return all_attentions

def save_attention_matrices(all_attentions):
    """Сохранение матриц внимания"""
    output_dir = Path("attentions")
    output_dir.mkdir(exist_ok=True)
    
    num_layers = len(all_attentions[0])
    num_heads = all_attentions[0][0].shape[0]
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            head_matrices = []
            for sample_idx in range(len(all_attentions)):
                attention_matrix = all_attentions[sample_idx][layer_idx][head_idx]
                head_matrices.append(attention_matrix.numpy())
            
            avg_attention = np.mean(head_matrices, axis=0)
            
            filename = f"attention_layer{layer_idx}_head{head_idx}.pt"
            torch.save(torch.tensor(avg_attention), output_dir / filename)
    
    print(f"Сохранено {num_layers} слоев, {num_heads} голов на слой")

def main():
    print("Загрузка модели...")
    model, tokenizer = load_model()
    
    print("Загрузка C4 примеров...")
    samples = load_c4_samples(tokenizer)
    
    print("Извлечение матриц внимания...")
    all_attentions = extract_attention_matrices(model, samples)
    
    print("Сохранение матриц...")
    save_attention_matrices(all_attentions)
    
    print("Готово!")

if __name__ == "__main__":
    main()
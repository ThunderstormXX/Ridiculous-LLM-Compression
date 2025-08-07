#!/usr/bin/env python3
"""
Запуск полного эксперимента по объединению attention heads
"""

import os
import subprocess
import argparse
import sys

def run_command(cmd, description):
    """Выполнить команду с описанием"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print(f"SUCCESS: {description} completed!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Run complete head merging experiment")
    parser.add_argument("--model_path", default="src/checkpoints/llama3.1-8b", 
                       help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.99, 
                       help="Cosine similarity threshold")
    parser.add_argument("--device", default="cuda:3", help="Device to use")
    parser.add_argument("--output_dir", default="polina_experiments/results", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Определяем пути к файлам
    similarity_file = os.path.join(args.output_dir, f"similarity_analysis_t{args.threshold}.json")
    merged_model_path = os.path.join(args.output_dir, f"merged_model_t{args.threshold}")
    merge_log_file = os.path.join(merged_model_path, "merge_log.json")
    visualization_dir = os.path.join(args.output_dir, f"visualization_t{args.threshold}")
    
    print(f"Starting head merging experiment with threshold {args.threshold}")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    
    # Шаг 1: Поиск похожих голов
    step1_cmd = f"cd /home/ThunderstormXX/Ridiculous-LLM-Compression && bash -c 'source .venv/bin/activate && python polina_experiments/step1_find_similar_heads.py --model_path {args.model_path} --threshold {args.threshold} --output_file {similarity_file} --device {args.device}'"
    
    if not run_command(step1_cmd, "Step 1: Finding similar heads"):
        return
    
    # Шаг 2: Объединение голов
    step2_cmd = f"cd /home/ThunderstormXX/Ridiculous-LLM-Compression && bash -c 'source .venv/bin/activate && python polina_experiments/step2_merge_heads.py --model_path {args.model_path} --similarity_file {similarity_file} --output_model_path {merged_model_path} --device {args.device}'"
    
    if not run_command(step2_cmd, "Step 2: Merging heads"):
        return
    
    # Шаг 3: Визуализация результатов
    step3_cmd = f"cd /home/ThunderstormXX/Ridiculous-LLM-Compression && bash -c 'source .venv/bin/activate && python polina_experiments/step3_visualize_results.py --original_model_path {args.model_path} --merged_model_path {merged_model_path} --merge_log_file {merge_log_file} --output_dir {visualization_dir} --device {args.device}'"
    
    if not run_command(step3_cmd, "Step 3: Visualizing results"):
        return
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Results saved in: {args.output_dir}")
    print(f"- Similarity analysis: {similarity_file}")
    print(f"- Merged model: {merged_model_path}")
    print(f"- Visualizations: {visualization_dir}")

if __name__ == "__main__":
    main()
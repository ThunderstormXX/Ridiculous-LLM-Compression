#!/usr/bin/env python3
"""
Главный скрипт для запуска всего эксперимента по анализу схожести голов внимания.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Запуск скрипта с обработкой ошибок"""
    print(f"\n{'='*60}")
    print(f"ЭТАП: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✅ {description} - ЗАВЕРШЕНО")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ОШИБКА в {description}:")
        print(f"Код возврата: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    
    return True

def main():
    print("🚀 ЗАПУСК ЭКСПЕРИМЕНТА ПО АНАЛИЗУ СХОЖЕСТИ ГОЛОВ ВНИМАНИЯ")
    print("Модель: TinyLlama")
    print("Датасет: C4")
    print("Метрика: L1 (средняя абсолютная разность)")
    
    if not run_script("extract_attention_matrices.py", 
                     "Извлечение матриц внимания"):
        print("❌ Эксперимент прерван на этапе извлечения матриц")
        return
    
    if not run_script("analyze_head_similarity.py", 
                     "Анализ схожести голов и создание визуализаций"):
        print("❌ Эксперимент прерван на этапе анализа схожести")
        return
    
    if not run_script("head_merging_experiment.py", 
                     "Эксперимент по объединению голов и измерению перплексии"):
        print("❌ Эксперимент прерван на этапе объединения голов")
        return
    
    print(f"\n{'='*60}")
    print("🎉 ЭКСПЕРИМЕНТ УСПЕШНО ЗАВЕРШЕН!")
    print(f"{'='*60}")
    print("\nРезультаты сохранены в:")
    print("📁 attentions/ - матрицы внимания")
    print("📄 similar_heads.txt - топ-10% похожих пар")
    print("📊 plots/ - тепловые карты и графики")
    print("📈 perplexity_trend.png - график перплексии")
    print("📋 perplexity_results.csv - данные эксперимента")

if __name__ == "__main__":
    main()
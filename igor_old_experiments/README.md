# 🦙 TinyLLaMA MMLU Benchmark & Distillation

Этот проект демонстрирует:

1. Загрузку модели [TinyLLaMA-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat)
2. Визуализацию архитектуры модели
3. Прогон на бенчмарке [MMLU](https://huggingface.co/datasets/cais/mmlu)
4. Простую дистилляцию — удаление части декодеров
5. Сравнение точности модели до и после дистилляции

## 📦 Установка

```bash
conda create -n tinyllama-env python=3.10 pip -y
pip install -r requirements.txt 
pip install numpy==1.24.3
pip install triton
pip install --upgrade bitsandbytes
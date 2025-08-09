# Новые скрипты для анализа голов внимания

## Описание
Эти скрипты предоставляют полный пайплайн для анализа похожести голов внимания и их влияния на перплексию модели.

## Файлы
- `compute_head_similarities.py` - вычисление матриц похожести
- `merge_heads_and_eval.py` - объединение голов и оценка перплексии  
- `plot_perplexities.py` - визуализация результатов
- `head_analysis_examples.md` - примеры использования

## Быстрый старт

### 1. Вычислить матрицы похожести для всех слоев
```bash
cd polina_experiments
python compute_head_similarities.py --model_key llama3.1-8b
```

### 2. Анализ конкретного слоя (например, слой 5)
```bash
python merge_heads_and_eval.py --model_key llama3.1-8b --layer 5
```

### 3. Анализ нескольких слоев
```bash
for layer in {0..10}; do
    python merge_heads_and_eval.py --model_key llama3.1-8b --layer $layer
done
```

### 4. Создание графиков
```bash
python plot_perplexities.py
```

## Результаты
- `logs/` - матрицы похожести (.npy) и результаты перплексии (.json)
- `figures/` - визуализации (heatmaps и графики)

## Доступные модели
- llama3.1-8b
- llama2-13b  
- mistral-7b
- phi2
- qwen-7b
- tinyllama
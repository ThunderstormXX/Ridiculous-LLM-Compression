# Эксперимент по объединению attention-голов

Этот эксперимент разделен на три отдельных шага для анализа и объединения похожих attention-голов в модели LLaMA.

## Структура эксперимента

### Шаг 1: Поиск похожих голов (`step1_find_similar_heads.py`)
- Вычисляет матрицы внимания для каждой пары голов по формуле: `a_ij = ReLU(h_i · Q_i · K_j · h_j)`
- Преобразует матрицы в векторную форму (flatten)
- Вычисляет косинусное сходство между векторизованными матрицами
- Находит пары голов с схожестью выше заданного порога
- Логирует результаты для каждого слоя

### Шаг 2: Объединение голов (`step2_merge_heads.py`)
- Загружает результаты анализа схожести из шага 1
- Объединяет похожие головы путем усреднения весов Q, K, V, O
- Сохраняет объединенную модель
- Создает лог процесса объединения

### Шаг 3: Оценка и визуализация (`step3_evaluate_and_visualize.py`)
- Загружает оригинальную и объединенную модели
- Использует датасет C4 для оценки (как в `scripts/igor_exps/iterative_pruning.py`)
- Вычисляет perplexity и accuracy до и после объединения
- Создает графики визуализации результатов

## Использование

### Запуск полного эксперимента
```bash
./polina_experiments/run_head_merging_experiment.sh --model_path src/checkpoints/llama3.1-8b --threshold 0.99
```

### Запуск отдельных шагов

#### Шаг 1: Поиск похожих голов
```bash
python polina_experiments/step1_find_similar_heads.py \
    --model_path src/checkpoints/llama3.1-8b \
    --threshold 0.99 \
    --output_dir polina_experiments/results \
    --device cpu
```

#### Шаг 2: Объединение голов
```bash
python polina_experiments/step2_merge_heads.py \
    --model_path src/checkpoints/llama3.1-8b \
    --similarity_file polina_experiments/results/similar_heads_threshold_0.99.json \
    --output_model_path polina_experiments/results/merged_model_threshold_0.99 \
    --device cpu
```

#### Шаг 3: Оценка и визуализация
```bash
python polina_experiments/step3_evaluate_and_visualize.py \
    --original_model_path src/checkpoints/llama3.1-8b \
    --merged_model_path polina_experiments/results/merged_model_threshold_0.99 \
    --output_dir polina_experiments/results \
    --device cpu
```

## Параметры

- `--model_path`: Путь к модели (по умолчанию: `src/checkpoints/llama3.1-8b`)
- `--threshold`: Порог косинусного сходства для объединения (по умолчанию: 0.99)
- `--output_dir`: Директория для результатов (по умолчанию: `polina_experiments/results`)
- `--device`: Устройство для вычислений (по умолчанию: `cpu`)

## Выходные файлы

После выполнения эксперимента в директории результатов будут созданы:

1. `similar_heads_threshold_X.json` - результаты анализа схожести
2. `merged_model_threshold_X/` - объединенная модель
3. `evaluation_results.json` - результаты оценки
4. `metrics_comparison.png` - график сравнения метрик
5. `merged_pairs_per_layer.png` - график количества объединенных пар по слоям
6. `metrics_change.png` - график изменения метрик

## Особенности реализации

### Вычисление косинусного сходства
- Используется формула внимания: `a_ij = ReLU(h_i · Q_i · K_j · h_j)`
- Матрицы внимания преобразуются в векторы для вычисления косинусного сходства
- Учитывается grouped attention для K и V матриц

### Объединение голов
- Веса Q и O усредняются для каждой пары похожих голов
- Для K и V учитывается архитектура grouped attention
- Сохраняются резервные копии оригинальных весов

### Оценка качества
- Используется датасет C4 (как в оригинальных экспериментах)
- Вычисляется perplexity и accuracy
- Создаются наглядные графики для сравнения результатов

## Пример запуска с разными порогами

```bash
# Высокий порог (строгое сходство)
./polina_experiments/run_head_merging_experiment.sh --threshold 0.995

# Средний порог
./polina_experiments/run_head_merging_experiment.sh --threshold 0.99

# Низкий порог (более мягкое сходство)
./polina_experiments/run_head_merging_experiment.sh --threshold 0.98
```
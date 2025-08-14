# Importance-Based Pruning Strategies

Этот модуль добавляет стратегии поиска слоев на основе их важности для прунинга моделей.

## Новые компоненты

### Модули
- `src/pruninghealing/layer_importance.py` - вычисление важности слоев из hidden states
- `src/pruninghealing/importance_strategies.py` - стратегии поиска на основе важности

### Стратегии
- `ImportanceBasedIterativeStrategy` - для итеративного прунинга
- `ImportanceBasedWindowStrategy` - для window прунинга

### Скрипты
- `scripts/compute_layer_importance.py` - вычисление важности слоев
- `scripts/igor_exps/bash/run_iterative_importance_pruning.sh` - итеративный прунинг с важностью
- `scripts/igor_exps/bash/run_window_importance_pruning.sh` - window прунинг с важностью

## Логика работы

### Для Iterative Pruner
Итеративный прунер удаляет слои по схеме: `start, start-2, start-4, ...`

Стратегия находит такой `start`, чтобы сумма важностей слоев `[start-num_layers*2+2, start-num_layers*2+4, ..., start-2, start]` была минимальной.

### Для Window Pruner  
Window прунер удаляет окно слоев `[start, start+1, ..., start+num_layers-1]`

Стратегия находит такой `start`, чтобы сумма важностей в окне `[start, start+num_layers-1]` была минимальной.

## Использование

### 1. Вычисление важности слоев
```bash
python scripts/compute_layer_importance.py \
    --hidden_path /path/to/hidden_states.json \
    --layer_type mlp \
    --output layer_importances.json
```

### 2. Итеративный прунинг с важностью
```bash
./scripts/igor_exps/bash/run_iterative_importance_pruning.sh \
    --model_path=src/checkpoints/llama3.1-8b \
    --workspace=./workspace \
    --num_layers=3 \
    --hidden_path=/path/to/hidden_states.json \
    --devices=0
```

### 3. Window прунинг с важностью
```bash
./scripts/igor_exps/bash/run_window_importance_pruning.sh \
    --model_path=src/checkpoints/llama3.1-8b \
    --workspace=./workspace \
    --window_size=3 \
    --hidden_path=/path/to/hidden_states.json \
    --devices=0
```

### 4. Использование через unified_pruning.py
```bash
python scripts/igor_exps/unified_pruning.py \
    --model_path src/checkpoints/llama3.1-8b \
    --method iterative \
    --strategy importance \
    --hidden_path /path/to/hidden_states.json \
    --num_layers 3
```

## Параметры

### Общие параметры
- `--strategy` - тип стратегии (`default` или `importance`)
- `--hidden_path` - путь к JSON файлу с hidden states (обязательно для `importance`)
- `--layer_type` - тип слоя для вычисления важности (`mlp`, `self_attn`, etc.)

### Для итеративного прунинга
- `--num_layers` - количество слоев для удаления
- `--start_layer` - стартовый слой (игнорируется при `importance` стратегии)

### Для window прунинга
- `--window_size` - размер окна для удаления

## Вывод стратегий

При использовании importance-based стратегий выводится:
1. Список всех слоев с их важностями (отсортированный)
2. Выбранный стартовый слой
3. Список слоев для удаления
4. Суммарная важность удаляемых слоев
5. Среднее значение важности по всем слоям
6. Дисперсия важности

Пример вывода:
```
Layer importances (sorted by importance):
  Layer 0: 0.123456
  Layer 1: 0.234567
  ...
Selected start layer: 24
Will remove layers: [24, 22, 20]
Total importance sum: 0.456789
Mean importance: 0.345678
Importance variance: 0.012345
```
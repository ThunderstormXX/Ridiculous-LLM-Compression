import os
import json
from .lora_utils import prune_layers, update_lora_layers

def get_num_layers(model):
    """
    Возвращает количество слоёв Llama, учитывая, что модель может быть
    PEFT-обёрткой или обычной.
    """
    # Спускаемся в глубину по известным атрибутам
    base = model
    if hasattr(base, "base_model"):
        base = base.base_model
    if hasattr(base, "model"):
        base = base.model
    if hasattr(base, "model"):
        base = base.model  # Обычно LlamaForCausalLM.model -> LlamaModel
    if hasattr(base, "layers"):
        return len(base.layers)
    raise AttributeError(f"Не удалось найти layers в объекте {model.__class__.__name__}")

def run_iterative_pruning_and_training(
    base_model,
    tokenizer,
    start_layer: int,
    num_layers_to_prune: int,
    train_fn,
    eval_fn,
    save_dir_prefix="trained_",
    log_path="log.json"
):
    current_peft_model = None
    current_target_modules = []

    # --- Изначально оцениваем базовую модель (без LoRA)
    print("[Init] Оценка perplexity базовой модели...")
    initial_perplexity = eval_fn(base_model, tokenizer, device=next(base_model.parameters()).device)
    print(f"  -> Initial perplexity: {initial_perplexity:.3f}")

    log = [{
        "iteration": 0,
        "removed_layer": None,
        "lora_applied_to": None,
        "remaining_layers": get_num_layers(base_model),
        "pre_train_perplexity": initial_perplexity,
        "post_train_perplexity": None,
        "saved_to": None
    }]

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    current_model = base_model

    for step in range(num_layers_to_prune):
        layer_index = start_layer + step
        print(f"\n=== Итерация {step+1}: слой {layer_index} ===")

        # 1. Удаляем слой из базовой модели
        current_model = prune_layers(current_model, [layer_index])
        print(f"  -> Слой {layer_index} удалён. Осталось: {get_num_layers(current_model)}")

        # 2. Обновляем LoRA — создаём PEFT-модель с новым target_modules списком
        current_peft_model, current_target_modules = update_lora_layers(
            base_model=current_model,
            current_peft_model=current_peft_model,
            current_target_modules=current_target_modules,
            new_layer_index=layer_index
        )
        print(f"  -> LoRA навешана на слой {layer_index}")

        # 3. Оценка perplexity ДО обучения
        print("[Step] Оценка perplexity ДО обучения...")
        perplexity_before = eval_fn(current_peft_model, tokenizer, device=next(current_peft_model.parameters()).device)
        print(f"  -> Pre-train perplexity: {perplexity_before:.3f}")

        # 4. Обучаем модель
        current_peft_model = train_fn(current_peft_model, tokenizer)

        # 5. Оценка perplexity ПОСЛЕ обучения
        print("[Step] Оценка perplexity ПОСЛЕ обучения...")
        perplexity_after = eval_fn(current_peft_model, tokenizer, device=next(current_peft_model.parameters()).device)
        print(f"  -> Post-train perplexity: {perplexity_after:.3f}")

        # 6. Сохраняем модель и токенизатор
        save_path = f"{save_dir_prefix}{step+1}"
        os.makedirs(save_path, exist_ok=True)
        current_peft_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # 7. Логируем итерацию
        step_log = {
            "iteration": step + 1,
            "removed_layer": layer_index,
            "lora_applied_to": layer_index,
            "remaining_layers": get_num_layers(current_model),
            "pre_train_perplexity": perplexity_before,
            "post_train_perplexity": perplexity_after,
            "saved_to": save_path
        }
        log.append(step_log)

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    return current_peft_model, log
import os
import json
from peft import LoraConfig, get_peft_model, TaskType
import torch

def get_num_layers(model):
    # Попытка получить количество слоёв с учётом возможной вложенности PEFT-обёртки
    try:
        # Если модель PEFT
        base_model = model.base_model.model.model
        return len(base_model.layers)
    except AttributeError:
        try:
            # Если модель LlamaForCausalLM без PEFT
            return len(model.model.layers)
        except AttributeError:
            raise RuntimeError(f"Не удалось определить число слоёв для модели класса {model.__class__.__name__}")

def prune_layers(model, layers_to_remove):
    # Удаляем слои в базовой модели (без PEFT)
    # Нужно получить чистую модель, удалить слои и обернуть заново в PEFT (если была)
    try:
        base_model = model.base_model.model.model
    except AttributeError:
        try:
            base_model = model.model
        except AttributeError:
            raise RuntimeError(f"Не удалось получить базовую модель для prune_layers у {model.__class__.__name__}")

    with torch.no_grad():
        kept_layers = [layer for idx, layer in enumerate(base_model.layers) if idx not in layers_to_remove]
        base_model.layers = torch.nn.ModuleList(kept_layers)
        base_model.config.num_hidden_layers = len(kept_layers)

    # Если исходная модель была обёрнута PEFT, вернуть её заново с сохранением LoRA
    if hasattr(model, "base_model"):
        # Пересоздавать PEFT-обёртку не нужно здесь — сделаем в основном цикле
        return model
    else:
        return model  # Обычная модель

def update_lora_layers(base_model, current_peft_model, current_target_modules, new_layer_index):
    # Генерируем новые target_modules для нового слоя
    new_targets = [
        f"model.layers.{new_layer_index}.self_attn.q_proj",
        f"model.layers.{new_layer_index}.self_attn.k_proj",
        f"model.layers.{new_layer_index}.self_attn.v_proj",
        f"model.layers.{new_layer_index}.self_attn.o_proj",
        f"model.layers.{new_layer_index}.mlp.gate_proj",
        f"model.layers.{new_layer_index}.mlp.up_proj",
        f"model.layers.{new_layer_index}.mlp.down_proj",
    ]

    # Добавляем новые, без дубликатов
    for t in new_targets:
        if t not in current_target_modules:
            current_target_modules.append(t)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=current_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Создаём новую PEFT-модель на базе base_model
    new_peft_model = get_peft_model(base_model, peft_config)

    # Копируем веса LoRA из старой PEFT-модели в новую, если есть
    if current_peft_model is not None:
        old_state = current_peft_model.state_dict()
        new_state = new_peft_model.state_dict()
        for k in new_state.keys():
            if k in old_state:
                new_state[k] = old_state[k]
        new_peft_model.load_state_dict(new_state)

    return new_peft_model, current_target_modules

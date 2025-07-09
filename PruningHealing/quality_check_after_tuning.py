# evaluate_quality_three_models.py

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

# Пути к моделям
base_model_path = "/home/user31/polina/Llama-3.2-3B-Instruct"  # базовая модель (model)
pruned_model_path = "/home/user31/igor/Llama-3.2-3B-Instruct-pruned"  # pruned
final_model_path = "/home/user31/igor/final_model"  # модель с LoRA

def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)

def load_pipeline(model_path, tokenizer, use_peft=False, base_model_path=None):
    if use_peft:
        # Загружаем базовую модель
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="cuda:0",
            torch_dtype="auto"
        )
        # Накладываем PEFT-адаптеры
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda:0",
            torch_dtype="auto"
        )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="cuda:0")

def generate_output(pipe, prompt):
    result = pipe(prompt, max_new_tokens=200, do_sample=False)
    return result[0]["generated_text"]

def main():
    tokenizer = load_tokenizer(base_model_path)

    base_pipe = load_pipeline(base_model_path, tokenizer)
    pruned_pipe = load_pipeline(pruned_model_path, tokenizer)
    final_pipe = load_pipeline(final_model_path, tokenizer, use_peft=True, base_model_path=pruned_model_path)

    prompts = [
        "Hello, tell me about the situation in Russia this year",
        "What are the benefits of pruning transformer models?",
        "Explain quantum computing in simple terms.",
        "Summarize the causes of World War I."
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i} ---\n{prompt}\n")
        print(f"[Base Model]:\n{generate_output(base_pipe, prompt)}\n")
        print(f"[Pruned Model]:\n{generate_output(pruned_pipe, prompt)}\n")
        print(f"[Final Model]:\n{generate_output(final_pipe, prompt)}\n")

if __name__ == "__main__":
    main()

# evaluate_quality.py

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

original_path = "/home/user31/polina/Llama-3.2-3B-Instruct"
pruned_path = "/home/user31/igor/Llama-3.2-3B-Instruct-pruned"

def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)

def load_model(model_path, tokenizer):
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
    tokenizer = load_tokenizer(original_path)
    original_pipe = load_model(original_path, tokenizer)
    pruned_pipe = load_model(pruned_path, tokenizer)

    prompts = [
        "Hello, tell me about the situation in Russia this year",
        "What are the benefits of pruning transformer models?",
        "Explain quantum computing in simple terms.",
        "Summarize the causes of World War I."
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i} ---\n{prompt}")
        print(f"\n[Original Model]:\n{generate_output(original_pipe, prompt)}")
        print(f"\n[Pruned Model]:\n{generate_output(pruned_pipe, prompt)}")

if __name__ == "__main__":
    main()

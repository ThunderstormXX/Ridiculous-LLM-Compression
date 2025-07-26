from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
import torch
torch.cuda.is_available()


DEVICE = model.device
DEVICE

from datasets import load_dataset
from transformers import pipeline
import evaluate

# Загружаем сабсет MMLU (например, "abstract_algebra")
dataset = load_dataset("cais/mmlu", "abstract_algebra", split="test")
metric = evaluate.load("accuracy")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="cuda:0")

def format_prompt(example):
    return f"Q: {example['question']}\nA:"

sequences = pipe(
    'The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01.',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    repetition_penalty=1.5,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
print(sequences)
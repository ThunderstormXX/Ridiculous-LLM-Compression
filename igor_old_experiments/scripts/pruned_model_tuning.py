from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import torch.nn as nn
from collections import Counter
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import time
import os

# === Загрузка обрезанной модели ===
model_path = "/home/user31/igor/Llama-3.2-3B-Instruct-pruned"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="cuda:3")


## ТОКЕНАЙЗЕР ЮЗАЕМ КАК У ИЗНАЧАЛЬНОЙ МОДЕЛИ
base_model_path = "/home/user31/polina/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)


# === Применяем LoRA ===
lora_config = LoraConfig(
    r=8,                      # размер low-rank
    lora_alpha=32,            # масштаб
    target_modules=["lm_head"],  # важно: имя слоя внутри mlp
    lora_dropout=0.05,        # dropout перед адаптером
    bias="none",              # bias не трогаем
    task_type=TaskType.CAUSAL_LM  # указываем тип задачи
)

# Применяем LoRA к модели
model = get_peft_model(model, lora_config)

# === Подготовка датасета ===
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

def format_dataset(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

tokenized_datasets = tokenized_datasets.map(format_dataset, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))


# === Кастомный Trainer с логами ===
class IterationLimitedTrainer(Trainer):
    def __init__(self, *args, max_iterations=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations
        self.writer = SummaryWriter(log_dir=self.args.logging_dir)
        self.start_time = time.time()

    def training_step(self, model, inputs, num_steps=None):
        if self.state.global_step >= self.max_iterations:
            self.control.should_training_stop = True
            return torch.tensor(0.0).to(self.args.device)

        loss = super().training_step(model, inputs, num_steps)

        if self.state.global_step % self.args.logging_steps == 0:
            perplexity = torch.exp(loss.detach())
            step_time = time.time() - self.start_time

            self.log({
                "loss": loss.item(),
                "perplexity": perplexity.item(),
                "iterations": self.state.global_step,
                "step_time": step_time
            })

            self.writer.add_scalar("train/loss", loss.item(), self.state.global_step)
            self.writer.add_scalar("train/perplexity", perplexity.item(), self.state.global_step)
            self.writer.add_scalar("train/learning_rate", self._get_learning_rate(), self.state.global_step)
            self.start_time = time.time()

        return loss


# === Аргументы обучения ===
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    do_eval=True,
    max_steps=3000,
    logging_dir="./logs",
    save_steps=500,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_strategy="steps",
    logging_steps=50,
    eval_steps=500,
    report_to="tensorboard",
    eval_strategy="steps",
    save_total_limit=2,
    gradient_accumulation_steps=2,
    warmup_steps=100,
)

# === Запуск обучения ===
trainer = IterationLimitedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_iterations=30000
)

try:
    trainer.train()
except KeyboardInterrupt:
    print("Обучение прервано пользователем")
finally:
    final_dir = "final_model"
    # Важно: сохраняем LoRA-модель через peft
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.writer.close()
    print(f"Модель и токенайзер сохранены в: {final_dir}")

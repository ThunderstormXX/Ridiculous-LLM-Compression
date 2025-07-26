from transformers import Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, load_from_disk
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import os

from peft import PeftModel


DATASET_CACHE_DIR = "./cached_dataset"  # можно сделать параметром

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


def preprocess_and_save_dataset(tokenizer, cache_dir=DATASET_CACHE_DIR):
    print(f"[Data] Загружаем и обрабатываем датасет...")

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def format_dataset(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_datasets = tokenized_datasets.map(format_dataset, batched=True)

    # Ограничим выборку
    tokenized_datasets = DatasetDict({
        "train": tokenized_datasets["train"].select(range(1000)),
        "validation": tokenized_datasets["validation"].select(range(100))
    })

    print(f"[Data] Сохраняем обработанный датасет в {cache_dir}...")
    tokenized_datasets.save_to_disk(cache_dir)
    return tokenized_datasets


def train(model, tokenizer):
    """
    Обучает модель с установленными LoRA-адаптерами.
    Возвращает обученную модель.
    """

    if os.path.exists(DATASET_CACHE_DIR):
        print(f"[Data] Загружаем токенизированный датасет из {DATASET_CACHE_DIR}...")
        tokenized_datasets = load_from_disk(DATASET_CACHE_DIR)
    else:
        tokenized_datasets = preprocess_and_save_dataset(tokenizer)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        do_eval=True,
        max_steps=500,
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
        trainer.writer.close()
        return model

# pruninghealing/trainer.py
import torch
from transformers import Trainer as HFTrainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter
import time
import os

class Trainer:
    def __init__(self, model, tokenizer, workspace_dir="./workspace"):
        self.model = model
        self.tokenizer = tokenizer
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
        
    def train(self, dataset, max_steps=500, learning_rate=2e-4, batch_size=2):
        """Train/fine-tune model on dataset"""
        output_dir = os.path.join(self.workspace_dir, "training_output")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=50,
            save_steps=max_steps,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            logging_dir=os.path.join(self.workspace_dir, "logs"),
            report_to="tensorboard"
        )
        
        trainer = IterationLimitedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset.train_dataset,
            eval_dataset=dataset.eval_dataset,
            tokenizer=self.tokenizer,
            max_iterations=max_steps
        )
        
        trainer.train()
        return self.model

class IterationLimitedTrainer(HFTrainer):
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
            self.log({
                "loss": loss.item(),
                "perplexity": perplexity.item(),
                "iterations": self.state.global_step
            })
            self.writer.add_scalar("train/loss", loss.item(), self.state.global_step)
            self.writer.add_scalar("train/perplexity", perplexity.item(), self.state.global_step)

        return loss
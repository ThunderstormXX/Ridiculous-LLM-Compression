# pruninghealing/dataset.py
from datasets import load_dataset, DatasetDict, load_from_disk
import os

class DatasetLoader:
    def __init__(self, tokenizer, cache_dir="./cached_dataset"):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.train_dataset = None
        self.eval_dataset = None
        
    def load_wikitext(self, max_length=512, train_size=1000, eval_size=100):
        """Load and tokenize WikiText-2 dataset"""
        if os.path.exists(self.cache_dir):
            tokenized_datasets = load_from_disk(self.cache_dir)
        else:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
            
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", 
                                    truncation=True, max_length=max_length)
            
            tokenized_datasets = dataset.map(tokenize_function, batched=True, 
                                           remove_columns=["text"])
            
            def format_dataset(examples):
                examples["labels"] = examples["input_ids"].copy()
                return examples
            
            tokenized_datasets = tokenized_datasets.map(format_dataset, batched=True)
            
            tokenized_datasets = DatasetDict({
                "train": tokenized_datasets["train"].select(range(train_size)),
                "validation": tokenized_datasets["validation"].select(range(eval_size))
            })
            
            tokenized_datasets.save_to_disk(self.cache_dir)
        
        self.train_dataset = tokenized_datasets["train"]
        self.eval_dataset = tokenized_datasets["validation"]
        return self
        
    def load_custom(self, dataset_path):
        """Load custom dataset from path"""
        if os.path.isdir(dataset_path):
            datasets = load_from_disk(dataset_path)
        else:
            datasets = load_dataset(dataset_path)
            
        self.train_dataset = datasets["train"]
        self.eval_dataset = datasets.get("validation", datasets.get("test"))
        return self
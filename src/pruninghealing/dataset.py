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
        
    def load_c4(self, max_length=256, train_size=1000, eval_size=100):
        """Load and tokenize C4 dataset"""
        cache_dir = self.cache_dir + "_c4"
        if os.path.exists(cache_dir):
            tokenized_datasets = load_from_disk(cache_dir)
        else:
            dataset = load_dataset('allenai/c4', 'en', streaming=True)
            
            # Take samples from streaming dataset
            train_samples = []
            eval_samples = []
            
            for i, sample in enumerate(dataset['train']):
                if len(train_samples) < train_size:
                    train_samples.append(sample)
                else:
                    break
                    
            for i, sample in enumerate(dataset['validation']):
                if len(eval_samples) < eval_size:
                    eval_samples.append(sample)
                else:
                    break
            
            from datasets import Dataset
            train_ds = Dataset.from_list(train_samples)
            eval_ds = Dataset.from_list(eval_samples)
            
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", 
                                    truncation=True, max_length=max_length)
            
            train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])
            eval_ds = eval_ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])
            
            def format_dataset(examples):
                examples["labels"] = examples["input_ids"].copy()
                return examples
            
            train_ds = train_ds.map(format_dataset, batched=True)
            eval_ds = eval_ds.map(format_dataset, batched=True)
            
            tokenized_datasets = DatasetDict({"train": train_ds, "validation": eval_ds})
            tokenized_datasets.save_to_disk(cache_dir)
        
        self.train_dataset = tokenized_datasets["train"]
        self.eval_dataset = tokenized_datasets["validation"]
        return self
        
    def load_custom(self, dataset_path, max_length=512):
        """Load custom dataset from path"""
        if os.path.isdir(dataset_path):
            datasets = load_from_disk(dataset_path)
        else:
            datasets = load_dataset(dataset_path)
        
        # Check if dataset needs tokenization
        sample = datasets["train"][0]
        if "input_ids" not in sample and "text" in sample:
            # Dataset needs tokenization
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", 
                                    truncation=True, max_length=max_length)
            
            # Remove all columns and tokenize
            columns_to_remove = datasets["train"].column_names
            datasets = datasets.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
            
            def format_dataset(examples):
                examples["labels"] = examples["input_ids"].copy()
                return examples
            
            datasets = datasets.map(format_dataset, batched=True)
            
        self.train_dataset = datasets["train"]
        self.eval_dataset = datasets.get("validation", datasets.get("test"))
        return self
# pruninghealing/utils.py
import torch
from datasets import load_dataset

def get_layers_base(model):
    """Universally extract base part of LLM model containing .layers attribute"""
    candidates = ['model', 'transformer', 'decoder', 'base_model']
    
    obj = model
    for _ in range(3):  # max 3 levels of nesting
        for cand in candidates:
            if hasattr(obj, cand):
                obj = getattr(obj, cand)
                if hasattr(obj, 'layers'):
                    return obj
        # Try unwrap (e.g., PeftModel -> base_model)
        if hasattr(obj, 'base_model'):
            obj = obj.base_model
        else:
            break
    return None

def get_model_layers(model):
    """Get number of layers in model"""
    base = get_layers_base(model)
    if base is None:
        raise RuntimeError(f"Cannot find layers in {model.__class__.__name__}")
    return len(base.layers)

def calculate_perplexity(model, tokenizer, dataset=None, dataset_name="allenai/c4", dataset_config="en", max_samples=100):
    """Calculate perplexity on evaluation dataset"""
    print(f"Starting perplexity calculation with {max_samples} samples...")
    model.eval()
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    
    # Use provided dataset or load from HuggingFace
    if dataset is None:
        if dataset_name == "allenai/c4":
            eval_dataset = load_dataset(dataset_name, dataset_config, split="validation", streaming=True)
            eval_samples = []
            for i, sample in enumerate(eval_dataset):
                if len(eval_samples) >= max_samples:
                    break
                eval_samples.append(sample)
            from datasets import Dataset
            eval_dataset = Dataset.from_list(eval_samples)
        else:
            eval_dataset = load_dataset(dataset_name, dataset_config, split="validation")
            eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    else:
        eval_dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    total_loss = 0
    total_tokens = 0
    
    print(f"Processing {len(eval_dataset)} samples...")
    with torch.no_grad():
        for i, example in enumerate(eval_dataset):
            if i % 5 == 0:
                print(f"Processing sample {i+1}/{len(eval_dataset)}...")
            
            # Handle different dataset formats
            text = None
            if "text" in example:
                text = example["text"]
            elif "input_ids" in example:
                # Already tokenized dataset
                inputs = {"input_ids": torch.tensor(example["input_ids"]).unsqueeze(0).to(device)}
                if "attention_mask" in example:
                    inputs["attention_mask"] = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)
                inputs["labels"] = inputs["input_ids"].clone()
            else:
                continue
                
            if text is not None:
                if not text.strip():
                    continue
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    if total_tokens == 0:
        print("Warning: No tokens processed!")
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"Perplexity calculation completed: {perplexity:.3f}")
    
    model.train()
    return perplexity

def load_model_and_tokenizer(model_path, device="auto"):
    """Load model and tokenizer from path"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use device_map to handle memory efficiently
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    elif device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    return model, tokenizer

def get_model_architecture(model):
    """Determine model architecture type"""
    model_type = model.config.model_type.lower()
    
    if "llama" in model_type:
        return "llama"
    elif "mistral" in model_type:
        return "mistral"
    elif "phi" in model_type:
        return "phi"
    elif "qwen" in model_type:
        return "qwen"
    else:
        return "unknown"

def safe_save_model(model, output_path):
    """Safely save model, handling meta device issues"""
    try:
        model.save_pretrained(output_path)
    except (NotImplementedError, RuntimeError) as e:
        if "meta tensor" in str(e) or "Cannot copy out of meta tensor" in str(e):
            print("Warning: Model has meta tensors, attempting to save with safe_serialization=False...")
            try:
                model.save_pretrained(output_path, safe_serialization=False)
            except Exception:
                print("Fallback: Moving model components to CPU before saving...")
                # Clear CUDA cache first
                torch.cuda.empty_cache()
                
                # For PEFT models
                if hasattr(model, 'base_model'):
                    if hasattr(model.base_model, 'model'):
                        model.base_model.model = model.base_model.model.to('cpu')
                    else:
                        model.base_model = model.base_model.to('cpu')
                else:
                    model = model.to('cpu')
                
                model.save_pretrained(output_path, safe_serialization=False)
        else:
            raise e
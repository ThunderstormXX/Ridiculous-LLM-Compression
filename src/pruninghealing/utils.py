# pruninghealing/utils.py
import torch
from datasets import load_dataset

def get_model_layers(model):
    """Get number of layers in model"""
    try:
        if hasattr(model, "base_model"):
            base = model.base_model.model.model
        else:
            base = model.model
        return len(base.layers)
    except AttributeError:
        raise RuntimeError(f"Cannot determine layer count for {model.__class__.__name__}")

def calculate_perplexity(model, tokenizer, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", max_samples=100):
    """Calculate perplexity on evaluation dataset"""
    model.eval()
    device = next(model.parameters()).device
    
    # Load evaluation dataset
    dataset = load_dataset(dataset_name, dataset_config, split="validation")
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for example in dataset:
            if not example["text"].strip():
                continue
                
            inputs = tokenizer(example["text"], return_tensors="pt", 
                             truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()
    return perplexity

def load_model_and_tokenizer(model_path, device="auto"):
    """Load model and tokenizer from path"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
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
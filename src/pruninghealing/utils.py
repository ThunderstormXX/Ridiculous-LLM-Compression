# pruninghealing/utils.py
import torch
from datasets import load_dataset
import math
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset

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

def calculate_perplexity(
    model,
    tokenizer,
    dataset,
    max_samples=None,
    max_length=512,
    normalized=True,
    device=None,
):
    """
    Calculate perplexity or normalized loss (divided by log vocab size).
    If normalized=True, returns normalized loss H_norm = H / log V (<=1 for better-than-random).
    Else returns perplexity PPL = exp(H).

    Returns single float.
    """
    import torch
    import math
    from tqdm import tqdm

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total_neg_log_likelihood = 0.0
    total_tokens = 0

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(tokenizer.get_vocab())
        except Exception:
            vocab_size = None

    pbar = tqdm(dataset, desc="Perplexity", unit="sample")

    with torch.no_grad():
        for example in pbar:
            if "text" in example:
                text = example["text"]
                if not text or not text.strip():
                    continue
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
            elif "input_ids" in example:
                ids = example["input_ids"]
                ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                inputs = {"input_ids": ids}
                if "attention_mask" in example:
                    am = torch.tensor(example["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
                    inputs["attention_mask"] = am
                inputs["labels"] = inputs["input_ids"].clone()
            else:
                continue

            outputs = model(**inputs)
            loss = outputs.loss
            seq_len = inputs["input_ids"].size(1)

            total_neg_log_likelihood += loss.item() * seq_len
            total_tokens += seq_len

            avg_loss_so_far = total_neg_log_likelihood / total_tokens
            if normalized and vocab_size:
                normalized_loss_so_far = avg_loss_so_far / math.log(vocab_size)
                pbar.set_postfix(
                    avg_loss=f"{avg_loss_so_far:.4f}",
                    norm_loss=f"{normalized_loss_so_far:.4f}",
                )
            else:
                ppl_so_far = math.exp(avg_loss_so_far)
                pbar.set_postfix(
                    avg_loss=f"{avg_loss_so_far:.4f}",
                    ppl=f"{ppl_so_far:.2f}",
                )

    model.train()

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_neg_log_likelihood / total_tokens

    if normalized and vocab_size:
        return avg_loss / math.log(vocab_size)  # normalized loss, <=1 for better-than-random
    else:
        return math.exp(avg_loss)  # perplexity





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
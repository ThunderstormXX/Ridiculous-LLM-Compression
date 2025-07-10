import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from memory_profiler import memory_usage
import gc

def load_model_and_tokenizer(model_path):
    """Load model with 4-bit quantization for memory efficiency"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32,
        # load_in_4bit=True  # Quantization for memory reduction
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def drop_layers(model, layers_to_drop):
    """Remove specified layers from the model"""
    if not layers_to_drop or layers_to_drop.lower() == "none":
        return model
    
    layers_to_drop = [int(x) for x in layers_to_drop.split(",")]
    kept_layers = [
        layer for i, layer in enumerate(model.model.layers) 
        if i not in layers_to_drop
    ]
    model.model.layers = torch.nn.ModuleList(kept_layers)
    model.config.num_hidden_layers = len(kept_layers)
    return model

def evaluate_memory(model, dataset_name, sample_size=10):
    """Test memory usage with sample inputs"""
    dataset = load_dataset(dataset_name, split=f"train[:{sample_size}]")
    mem_before = torch.cuda.memory_allocated()
    
    def inference():
        with torch.no_grad():
            for text in dataset["text"]:
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                _ = model.generate(**inputs, max_new_tokens=20)
    
    mem_usage = memory_usage(inference)
    mem_after = torch.cuda.memory_allocated()
    
    return {
        "peak_memory": max(mem_usage),
        "memory_saved": mem_before - mem_after
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to LLaMA model")
    parser.add_argument("dataset_name", help="HuggingFace dataset name")
    parser.add_argument("layers_to_drop", help="Comma-separated layer indices or 'none'")
    args = parser.parse_args()

    # Load and modify model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    original_size = sum(p.numel() for p in model.parameters())
    
    print(f"\nOriginal model layers: {len(model.model.layers)}")
    model = drop_layers(model, args.layers_to_drop)
    print(f"Modified model layers: {len(model.model.layers)}")

    # Evaluate memory
    print("\nEvaluating memory usage...")
    metrics = evaluate_memory(model, args.dataset_name)
    
    print("\nResults:")
    print(f"- Peak memory during inference: {metrics['peak_memory']:.2f} MB")
    print(f"- Estimated memory saved: {metrics['memory_saved'] / 1e6:.2f} MB")
    print(f"- Layers removed: {len(model.model.layers) - len(model.model.layers)}")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

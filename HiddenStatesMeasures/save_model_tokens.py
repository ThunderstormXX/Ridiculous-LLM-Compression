import argparse
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from memory_profiler import memory_usage
import gc
from datetime import datetime

def load_model_and_tokenizer(model_path):
    """Load model with 4-bit quantization for memory efficiency"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
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

def generate_and_save(model, tokenizer, dataset_name, output_file, sample_size=5):
    """Generate text and save results to JSON"""
    dataset = load_dataset(dataset_name, split=f"train[:{sample_size}]")
    results = []
    
    for i, text in enumerate(dataset["text"]):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7
            )
        
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            "input": text,
            "output": decoded,
            "timestamp": datetime.now().isoformat()
        })
    
    # Save to JSON with pretty formatting
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return len(results)

def evaluate_memory(model, dataset_name):
    """Measure memory usage during generation"""
    def generation_wrapper():
        generate_and_save(model, tokenizer, dataset_name, "/dev/null", sample_size=3)
    
    return max(memory_usage(generation_wrapper))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA Memory Optimizer with Text Generation")
    parser.add_argument("model_path", help="Path to LLaMA model")
    parser.add_argument("dataset_name", help="HuggingFace dataset name")
    parser.add_argument("layers_to_drop", help="Comma-separated layer indices or 'none'")
    parser.add_argument("--output", default="generated_tokens.json", help="Output JSON file path")
    args = parser.parse_args()

    # Initialize
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model = drop_layers(model, args.layers_to_drop)
    
    # Generate and save
    print(f"\nGenerating text samples (saving to {args.output})...")
    num_samples = generate_and_save(model, tokenizer, args.dataset_name, args.output)
    
    # Evaluate
    print("\nEvaluating memory usage...")
    peak_mem = evaluate_memory(model, args.dataset_name)
    
    print("\nResults:")
    print(f"- Generated {num_samples} samples")
    print(f"- Peak memory usage: {peak_mem:.2f} MB")
    print(f"- Output saved to: {Path(args.output).absolute()}")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

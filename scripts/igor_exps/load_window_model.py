#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_window_model(model_name="llama3.1-8b", device="auto"):
    """Load window pruned model with adapter"""
    
    base_path = f"src/checkpoints/{model_name}_p_window_base"
    adapter_path = f"src/checkpoints/{model_name}_p_window"
    
    print(f"Loading base model from: {base_path}")
    print(f"Loading adapter from: {adapter_path}")
    
    # Handle device mapping
    if device.isdigit():
        device_map = f"cuda:{device}"
    elif device == "auto":
        device_map = "auto"
    else:
        device_map = device
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    
    # Load base model
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            device_map=device_map
        )
    
    # Load model with adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Base model layers: {len(base_model.model.layers) if hasattr(base_model.model, 'layers') else 'Unknown'}")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Load window pruned model")
    parser.add_argument("--model_name", default="llama3.1-8b", help="Model name")
    parser.add_argument("--device", default="auto", help="Device")
    parser.add_argument("--test_generation", action="store_true", help="Test text generation")
    
    args = parser.parse_args()
    
    model, tokenizer = load_window_model(args.model_name, args.device)
    
    if args.test_generation:
        print("\nTesting generation...")
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20, do_sample=False)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()
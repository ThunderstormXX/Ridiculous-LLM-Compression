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
    
    # Check if adapter exists
    if os.path.exists(adapter_path):
        print(f"Loading base model from: {base_path}")
        print(f"Loading adapter from: {adapter_path}")
        use_adapter = True
    else:
        print(f"Loading merged model from: {base_path}")
        use_adapter = False
    
    # Handle device mapping
    if device.isdigit():
        device_map = f"cuda:{device}"
    elif device == "auto":
        device_map = "auto"
    else:
        device_map = device
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    
    # Load model
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            device_map=device_map
        )
    
    if use_adapter:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"Model with adapter loaded successfully!")
    else:
        model = base_model
        print(f"Merged model loaded successfully!")
    
    print(model)
    print(f"Model type: {type(model)}")
    print(f"Model layers: {len(model.model.layers if hasattr(model, 'model') else model.model.layers) if hasattr(model.model if hasattr(model, 'model') else model, 'layers') else 'Unknown'}")
    
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
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20, do_sample=False)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()
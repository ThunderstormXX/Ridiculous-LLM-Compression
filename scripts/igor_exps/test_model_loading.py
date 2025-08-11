#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from load_iterative_model import load_iterative_model
from load_window_model import load_window_model

def test_model_loading():
    """Test loading both types of pruned models"""
    
    model_name = "llama3.1-8b"
    
    print("=" * 50)
    print("Testing Iterative Model Loading")
    print("=" * 50)
    
    try:
        iter_model, iter_tokenizer = load_iterative_model(model_name)
        print("✓ Iterative model loaded successfully")
        
        # Test generation
        inputs = iter_tokenizer("Hello", return_tensors="pt")
        outputs = iter_model.generate(**inputs, max_length=10, do_sample=False)
        text = iter_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation test: {text}")
        
    except Exception as e:
        print(f"✗ Failed to load iterative model: {e}")
    
    print("\n" + "=" * 50)
    print("Testing Window Model Loading")
    print("=" * 50)
    
    try:
        window_model, window_tokenizer = load_window_model(model_name)
        print("✓ Window model loaded successfully")
        
        # Test generation
        inputs = window_tokenizer("Hello", return_tensors="pt")
        outputs = window_model.generate(**inputs, max_length=10, do_sample=False)
        text = window_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation test: {text}")
        
    except Exception as e:
        print(f"✗ Failed to load window model: {e}")

if __name__ == "__main__":
    test_model_loading()
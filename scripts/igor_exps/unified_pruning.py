# scripts/igor_exps/unified_pruning.py
import argparse
import os
import sys
import torch
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.pruninghealing import Trainer, DatasetLoader, IterativePruner, WindowPruner
from src.pruninghealing.prune import DefaultIterativeStrategy, DefaultWindowStrategy
from src.pruninghealing.importance_strategies import ImportanceBasedIterativeStrategy, ImportanceBasedWindowStrategy
from src.pruninghealing.layer_importance import compute_layer_importances
from src.pruninghealing.utils import load_model_and_tokenizer, calculate_perplexity, get_model_layers, safe_save_model
from src.pruninghealing.logger import Logger

def generate_hidden_states(model, tokenizer, output_path, cached_dataset, layer_type, max_samples, max_tokens):
    """Generate hidden states for importance calculation"""
    import numpy as np
    
    print(f"Using cached dataset for hidden states generation...")
    dataset = cached_dataset['validation'].select(range(min(max_samples, len(cached_dataset['validation']))))
    
    hidden_states = {layer_type: {}}
    num_layers = get_model_layers(model)
    
    # Initialize storage
    for layer_idx in range(num_layers):
        hidden_states[layer_type][str(layer_idx)] = {
            'hidden_states': [],
            'shapes': []
        }
    
    model.eval()
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
            
            text = example.get('text', str(example))
            inputs = tokenizer(text, return_tensors="pt", max_length=max_tokens, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Hook to capture hidden states
            layer_outputs = {}
            
            def make_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    layer_outputs[layer_idx] = output.cpu().numpy()
                return hook
            
            # Register hooks
            hooks = []
            for layer_idx in range(num_layers):
                if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    layer = model.model.layers[layer_idx]
                elif hasattr(model, 'layers'):
                    layer = model.layers[layer_idx]
                else:
                    continue
                    
                if layer_type == 'mlp' and hasattr(layer, 'mlp'):
                    hook = layer.mlp.register_forward_hook(make_hook(layer_idx))
                elif layer_type == 'self_attn' and hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(make_hook(layer_idx))
                else:
                    hook = layer.register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Store hidden states
            for layer_idx in range(num_layers):
                if layer_idx in layer_outputs:
                    hidden_state = layer_outputs[layer_idx]
                    hidden_states[layer_type][str(layer_idx)]['hidden_states'].append(hidden_state.flatten().tolist())
                    hidden_states[layer_type][str(layer_idx)]['shapes'].append(hidden_state.shape)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{max_samples} samples")
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(hidden_states, f)
    
    print(f"Hidden states saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified pruning and healing experiment")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--method", choices=["iterative", "window"], required=True, help="Pruning method")
    parser.add_argument("--max_steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--skip_training", action="store_true", help="Skip training (debug mode)")
    parser.add_argument("--skip_perplexity", action="store_true", help="Skip perplexity calculations (debug mode)")
    
    # Strategy parameters
    parser.add_argument("--strategy", choices=["default", "importance"], default="default", help="Search strategy")
    parser.add_argument("--hidden_path", type=str, help="Path to hidden states JSON (optional for importance strategy)")
    parser.add_argument("--layer_type", type=str, default="mlp", help="Layer type for importance calculation")
    parser.add_argument("--max_samples", type=int, default=80, help="Max samples for hidden states")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens for hidden states")
    
    # Method-specific parameters
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers to prune (iterative)")
    parser.add_argument("--start_layer", type=int, default=0, help="Starting layer (iterative)")
    parser.add_argument("--window_size", type=int, default=3, help="Window size (window)")
    
    args = parser.parse_args()
    
    # Create experiment directory
    exp_dir = os.path.join(args.workspace, f"{args.method}_pruning")
    os.makedirs(exp_dir, exist_ok=True)
    run_num = 1
    while os.path.exists(os.path.join(exp_dir, f"run_{run_num}")):
        run_num += 1
    run_dir = os.path.join(exp_dir, f"run_{run_num}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("Loading model...")
    device = args.device
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=device)
    print(f"Model loaded: {get_model_layers(model)} layers")
    
    # Load cached dataset early for importance calculation
    print("Loading cached dataset...")
    from datasets import load_from_disk
    cached_dataset_path = os.path.join(os.path.dirname(__file__), '../../cached_dataset')
    raw_dataset = load_from_disk(cached_dataset_path)
    
    # Create search strategy
    if args.strategy == "importance":
        if not args.hidden_path:
            print("Generating hidden states from cached dataset...")
            hidden_path = os.path.join(run_dir, "hidden_states.json")
            generate_hidden_states(model, tokenizer, hidden_path, raw_dataset, args.layer_type, args.max_samples, args.max_tokens)
            args.hidden_path = hidden_path
        
        print(f"Computing layer importances from {args.hidden_path}...")
        importances = compute_layer_importances(args.hidden_path, args.layer_type)
        
        if args.method == "iterative":
            search_strategy = ImportanceBasedIterativeStrategy(importances)
        else:
            search_strategy = ImportanceBasedWindowStrategy(importances)
    else:
        if args.method == "iterative":
            search_strategy = DefaultIterativeStrategy()
        else:
            search_strategy = DefaultWindowStrategy()
    
    # Skip dataset loading in debug mode
    if args.skip_training and args.skip_perplexity:
        print("Skipping dataset processing (debug mode)")
        baseline_ppl = 0.0
        dataset_obj = None
    else:
        print("Calculating baseline perplexity...")
        if not args.skip_perplexity:
            baseline_ppl = calculate_perplexity(model, tokenizer, dataset=raw_dataset['validation'])
            print(f"Baseline perplexity: {baseline_ppl:.3f}")
        else:
            baseline_ppl = 0.0
            print("Skipping baseline perplexity calculation (debug mode)")
        
        # Tokenize dataset for training
        if not args.skip_training:
            print("Tokenizing dataset for training...")
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
            tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset['train'].column_names)
            
            def format_dataset(examples):
                examples["labels"] = examples["input_ids"].copy()
                return examples
            
            tokenized_dataset = tokenized_dataset.map(format_dataset, batched=True)
        else:
            tokenized_dataset = None
        
        # Create simple dataset object for training
        class SimpleDataset:
            def __init__(self, train_data, eval_data):
                self.train_dataset = train_data
                self.eval_dataset = eval_data
        
        dataset_obj = SimpleDataset(tokenized_dataset['train'] if tokenized_dataset else None, raw_dataset['validation'] if not args.skip_perplexity else None)
    
    # Initialize components
    trainer = Trainer(model, tokenizer, run_dir)
    logger = Logger(run_dir)
    
    # Log baseline
    logger.log_step({
        "step": 0,
        "action": "baseline",
        "method": args.method,
        "strategy": args.strategy,
        "layers_total": get_model_layers(model),
        "perplexity": baseline_ppl
    })
    
    # Run pruning method
    if args.method == "iterative":
        pruner = IterativePruner(model, tokenizer, run_dir)
        final_model = pruner.prune_and_heal(
            dataset=dataset_obj,
            trainer=trainer,
            logger=logger,
            start_layer=args.start_layer,
            num_layers=args.num_layers,
            max_steps=args.max_steps,
            search_strategy=search_strategy,
            skip_training=args.skip_training,
            skip_perplexity=args.skip_perplexity
        )
        method_suffix = "iter"
    else:  # window
        pruner = WindowPruner(model, tokenizer, run_dir)
        final_model = pruner.prune_and_heal(
            dataset=dataset_obj,
            trainer=trainer,
            logger=logger,
            window_size=args.window_size,
            max_steps=args.max_steps,
            search_strategy=search_strategy,
            skip_training=args.skip_training,
            skip_perplexity=args.skip_perplexity
        )
        method_suffix = "window"
    
    # Save final model
    model_name = os.path.basename(args.model_path.rstrip('/'))
    debug_suffix = "_debug" if args.skip_training else ""
    strategy_suffix = f"_{args.strategy}" if args.strategy != "default" else ""
    
    # Define paths for base model and adapter
    base_output_path = f"src/checkpoints/{model_name}_p_{method_suffix}_base{strategy_suffix}{debug_suffix}"
    adapter_output_path = f"src/checkpoints/{model_name}_p_{method_suffix}{strategy_suffix}{debug_suffix}"
    os.makedirs("src/checkpoints", exist_ok=True)
    
    print(f"\nSaving models:")
    print(f"Base model: {base_output_path}")
    print(f"Adapter: {adapter_output_path}")
    
    # Save models
    from peft import PeftModel
    print(f"Final model type: {type(final_model)}")
    print(f"Is PeftModel: {isinstance(final_model, PeftModel)}")
    
    if isinstance(final_model, PeftModel):
        print('Saving PEFT model...')
        # Merge and save pruned model as base
        merged_model = final_model.merge_and_unload()
        merged_model.save_pretrained(base_output_path)
        tokenizer.save_pretrained(base_output_path)
        
        # Save adapter separately
        final_model.save_pretrained(adapter_output_path, save_embedding_layers=False)
        
        print(f"Merged base model: {base_output_path}")
        print(f"Adapter: {adapter_output_path}")
    else:
        print('Model is not PEFT, saving directly...')
        final_model.save_pretrained(base_output_path)
        tokenizer.save_pretrained(base_output_path)
        
    if args.skip_training:
        print(f"\n[DEBUG] Base model saved: {base_output_path}")
        print(f"[DEBUG] Adapter saved: {adapter_output_path}")
        print(f"[DEBUG] Model architecture: {final_model}")
    
    print("\nExperiment completed!")
    print(f"Results saved to: {run_dir}")
    print(f"Base model saved to: {base_output_path}")
    print(f"Adapter saved to: {adapter_output_path}")

if __name__ == "__main__":
    main()
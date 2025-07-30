# scripts/iterative_pruning.py
import argparse
import sys
sys.path.append('..')

from src.pruninghealing import Trainer, DatasetLoader, IterativePruner
from src.pruninghealing.utils import load_model_and_tokenizer, calculate_perplexity, get_model_layers
from src.pruninghealing.logger import Logger

def test_model_quality(model, tokenizer, prompt="What is the capital of France?"):
    """Test model response quality"""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    parser = argparse.ArgumentParser(description="Run iterative pruning experiment")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers to prune")
    parser.add_argument("--start_layer", type=int, default=0, help="Starting layer index")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total training budget (steps)")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.workspace, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    print(f"Model loaded: {get_model_layers(model)} layers")
    
    # Load dataset
    print("Loading C4 dataset...")
    dataset_loader = DatasetLoader(tokenizer)
    dataset_loader.load_c4(train_size=500, eval_size=50)
    
    # Calculate baseline
    print("Calculating baseline perplexity...")
    baseline_ppl = calculate_perplexity(model, tokenizer, max_samples=20)
    print(f"Baseline perplexity: {baseline_ppl:.3f}")
    
    # Initialize components
    pruner = IterativePruner(model, tokenizer, args.workspace)
    trainer = Trainer(model, tokenizer, args.workspace)
    logger = Logger(args.workspace)
    
    # Log baseline
    logger.log_step({
        "step": 0,
        "action": "baseline",
        "layers_remaining": get_model_layers(model),
        "perplexity": baseline_ppl
    })
    
    # Calculate steps per iteration
    steps_per_iter = args.max_steps // args.num_layers
    total_steps_used = 0
    
    # Run iterative pruning
    current_model = model
    
    for step in range(args.num_layers):
        layer_idx = args.start_layer + step
        print(f"\n=== Step {step+1}: Processing layer {layer_idx} ===")
        
        # Remove layer
        current_model = pruner._remove_layer(current_model, layer_idx)
        layers_remaining = get_model_layers(current_model)
        print(f"Layers remaining: {layers_remaining}")
        
        # Test after pruning
        ppl_after_prune = calculate_perplexity(current_model, tokenizer, max_samples=20)
        print(f"Perplexity after pruning: {ppl_after_prune:.3f}")
        
        # Apply LoRA
        print(f"Applying LoRA to layer {layer_idx}...")
        current_model = pruner._apply_lora(current_model, layer_idx)
        
        # Enable gradients for LoRA parameters
        for param in current_model.parameters():
            if param.requires_grad:
                param.requires_grad_(True)
        
        # Train model with budget
        remaining_budget = args.max_steps - total_steps_used
        current_steps = min(steps_per_iter, remaining_budget)
        print(f"Training model ({current_steps} steps, {total_steps_used}/{args.max_steps} used)...")
        
        # Clear cache before training
        torch.cuda.empty_cache()
        
        trainer.model = current_model
        current_model = trainer.train(dataset_loader, max_steps=current_steps)
        total_steps_used += current_steps
        
        # Clear cache after training
        torch.cuda.empty_cache()
        
        # Test after training
        ppl_after_train = calculate_perplexity(current_model, tokenizer, max_samples=20)
        print(f"Perplexity after training: {ppl_after_train:.3f}")
        
        # Log step with training info
        logger.log_step({
            "action": "prune", 
            "step": step + 1, 
            "layer": layer_idx, 
            "ppl": ppl_after_prune,
            "layers_remaining": layers_remaining
        })
        logger.log_step({
            "action": "train", 
            "step": step + 1, 
            "layer": layer_idx, 
            "ppl": ppl_after_train,
            "training_steps": current_steps,
            "total_steps_used": total_steps_used,
            "budget_remaining": args.max_steps - total_steps_used
        })
        
        print(f"Step {step+1} completed! Budget used: {total_steps_used}/{args.max_steps}")
        
        if total_steps_used >= args.max_steps:
            print("Training budget exhausted!")
            break
    
    # Save final pruned model
    import os
    model_name = os.path.basename(args.model_path.rstrip('/'))
    output_path = f"src/checkpoints/{model_name}_p_iter"
    os.makedirs("src/checkpoints", exist_ok=True)
    
    print(f"\nSaving pruned model to: {output_path}")
    current_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("\nIterative pruning completed!")
    print(f"Final model saved to: {output_path}")

if __name__ == "__main__":
    import torch
    main()
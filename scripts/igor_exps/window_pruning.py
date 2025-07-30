# scripts/window_pruning.py
import argparse
import torch
import os
import json
from datetime import datetime
from src.pruninghealing.utils import load_model_and_tokenizer, get_model_layers, calculate_perplexity
from src.pruninghealing.prune import WindowPruner
from src.pruninghealing import Trainer, DatasetLoader

def log_step(log_file, data):
    """Log experiment step"""
    data["timestamp"] = datetime.now().isoformat()
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log = json.load(f)
    else:
        log = []
    
    log.append(data)
    
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Prune window layers and fine-tune")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices to remove")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total training budget (steps)")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    os.makedirs(args.workspace, exist_ok=True)
    log_file = os.path.join(args.workspace, "experiment_log.json")
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    # Log baseline
    baseline_ppl = calculate_perplexity(model, tokenizer, max_samples=20)
    print(f"Baseline perplexity: {baseline_ppl:.3f}")
    
    log_step(log_file, {
        "step": 0,
        "action": "baseline",
        "layers_remaining": get_model_layers(model),
        "perplexity": baseline_ppl,
        "model_path": args.model_path
    })
    
    # Parse layer indices
    layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    
    print(f"Original model layers: {get_model_layers(model)}")
    print(f"Removing layers: {layer_indices}")
    
    # Create window pruner and remove layers
    pruner = WindowPruner(model, tokenizer, args.workspace)
    
    # Remove window of layers
    model = pruner.prune_window(layer_indices)
    
    # Enable gradients for LoRA parameters
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad_(True)
    
    layers_remaining = get_model_layers(model)
    print(f"Remaining layers: {layers_remaining}")
    
    # Test after pruning
    ppl_after_prune = calculate_perplexity(model, tokenizer, max_samples=20)
    print(f"Perplexity after pruning: {ppl_after_prune:.3f}")
    
    log_step(log_file, {
        "step": 1,
        "action": "prune",
        "layers_removed": layer_indices,
        "layers_remaining": layers_remaining,
        "perplexity": ppl_after_prune
    })
    
    # Save pruned model to checkpoints
    model_name = os.path.basename(args.model_path.rstrip('/'))
    pruned_path = f"src/checkpoints/{model_name}_p_window"
    os.makedirs("src/checkpoints", exist_ok=True)
    
    model.save_pretrained(pruned_path)
    tokenizer.save_pretrained(pruned_path)
    print(f"Pruned model saved to: {pruned_path}")
    
    # Clear memory and reload pruned model
    del model, pruner
    torch.cuda.empty_cache()
    
    print("Reloading pruned model for fine-tuning...")
    model, tokenizer = load_model_and_tokenizer(pruned_path, device=args.device)
    
    # Enable gradients for all trainable parameters after reload
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad_(True)
    
    # Load dataset for fine-tuning
    print("Loading C4 dataset...")
    dataset_loader = DatasetLoader(tokenizer)
    dataset_loader.load_c4(train_size=500, eval_size=50)
    
    # Fine-tune with full budget
    print(f"Fine-tuning with {args.max_steps} steps...")
    
    # Clear cache before training
    torch.cuda.empty_cache()
    
    trainer = Trainer(model, tokenizer, args.workspace)
    model = trainer.train(dataset_loader, max_steps=args.max_steps)
    
    # Clear cache after training
    torch.cuda.empty_cache()
    
    # Test after training
    final_ppl = calculate_perplexity(model, tokenizer, max_samples=20)
    print(f"Final perplexity: {final_ppl:.3f}")
    
    log_step(log_file, {
        "step": 2,
        "action": "train",
        "perplexity": final_ppl,
        "training_steps": args.max_steps,
        "total_steps_used": args.max_steps
    })
    
    # Save fine-tuned model
    finetuned_path = f"src/checkpoints/{model_name}_p_window_finetuned"
    model.save_pretrained(finetuned_path)
    tokenizer.save_pretrained(finetuned_path)
    print(f"Fine-tuned model saved to: {finetuned_path}")
    
    print(f"Experiment completed! Log saved to: {log_file}")

if __name__ == "__main__":
    main()
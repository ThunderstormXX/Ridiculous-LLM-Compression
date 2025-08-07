# scripts/igor_exps/window_pruning.py
import argparse
import torch
import os
import sys
import json
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
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
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--window_size", type=int, default=3, help="Window size for layer removal")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total training budget (steps)")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Create experiment directory
    exp_dir = os.path.join(args.workspace, "window_pruning")
    os.makedirs(exp_dir, exist_ok=True)
    run_num = 1
    while os.path.exists(os.path.join(exp_dir, f"run_{run_num}")):
        run_num += 1
    run_dir = os.path.join(exp_dir, f"run_{run_num}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "experiment_log.json")
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    # Load dataset for evaluation
    print("Loading C4 dataset...")
    dataset_loader = DatasetLoader(tokenizer)
    dataset_loader.load_c4(train_size=500, eval_size=50)
    
    # Log baseline
    baseline_ppl = calculate_perplexity(model, tokenizer, dataset=dataset_loader.eval_dataset, max_samples=20)
    print(f"Baseline perplexity: {baseline_ppl:.3f}")
    
    log_step(log_file, {
        "step": 0,
        "action": "baseline",
        "layers_remaining": get_model_layers(model),
        "perplexity": baseline_ppl,
        "model_path": args.model_path
    })
    
    # Find unimportant layers
    from src.pruninghealing.prune import WindowPruner as SearchPruner
    search_pruner = SearchPruner(model, tokenizer, run_dir)
    layer_indices, _ = search_pruner.find_unimportant_window(args.window_size)
    
    print(f"Original model layers: {get_model_layers(model)}")
    print(f"Removing layers: {layer_indices}")
    
    # 1) Remove layers
    pruner = WindowPruner(model, tokenizer, run_dir)
    base_model = pruner._get_base_model(model)
    with torch.no_grad():
        layers = [layer for i, layer in enumerate(base_model.layers) if i not in layer_indices]
        base_model.layers = torch.nn.ModuleList(layers)
        base_model.config.num_hidden_layers = len(layers)
    
    layers_remaining = get_model_layers(model)
    print(f"Remaining layers: {layers_remaining}")
    
    # Test after pruning
    ppl_after_prune = calculate_perplexity(model, tokenizer, dataset=dataset_loader.eval_dataset, max_samples=20)
    print(f"Perplexity after pruning: {ppl_after_prune:.3f}")
    
    # 2) Apply LoRA to last few MLP layers
    last_layers = min(3, layers_remaining)
    target_modules = []
    for i in range(layers_remaining - last_layers, layers_remaining):
        target_modules.extend([f"model.layers.{i}.mlp.gate_proj", f"model.layers.{i}.mlp.down_proj", f"model.layers.{i}.mlp.up_proj"])
    
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    
    # 3) Save initial pruned model (before training)
    model_name = os.path.basename(args.model_path.rstrip('/'))
    pruned_path = f"src/checkpoints/{model_name}_p_window"
    os.makedirs("src/checkpoints", exist_ok=True)
    
    model.save_pretrained(pruned_path)
    tokenizer.save_pretrained(pruned_path)
    print(f"Pruned model saved to: {pruned_path}")
    
    log_step(log_file, {
        "step": 1,
        "action": "prune",
        "layers_removed": layer_indices,
        "layers_remaining": layers_remaining,
        "perplexity": ppl_after_prune
    })
    

    
    # 4) Train model and save trained version
    print(f"Fine-tuning with {args.max_steps} steps...")
    torch.cuda.empty_cache()
    
    trainer = Trainer(model, tokenizer, run_dir)
    model = trainer.train(dataset_loader, max_steps=args.max_steps)
    torch.cuda.empty_cache()
    
    # Test after training
    final_ppl = calculate_perplexity(model, tokenizer, dataset=dataset_loader.eval_dataset, max_samples=20)
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
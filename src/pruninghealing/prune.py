# pruninghealing/prune.py
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import json
import os
from .utils import get_model_layers, calculate_perplexity
import os

class LayerSearchStrategy:
    """Base class for layer search strategies"""
    def find_start_layer(self, model, tokenizer, num_layers):
        raise NotImplementedError

class DefaultIterativeStrategy(LayerSearchStrategy):
    """Default strategy for iterative pruning - returns constant layer 19"""
    def find_start_layer(self, model, tokenizer, num_layers):
        return 19

class DefaultWindowStrategy(LayerSearchStrategy):
    """Default strategy for window pruning - returns constant layer 3"""
    def find_start_layer(self, model, tokenizer, num_layers):
        return 19

class IterativePruner:
    def __init__(self, model, tokenizer, workspace_dir="./workspace"):
        self.model = model
        self.tokenizer = tokenizer
        self.workspace_dir = workspace_dir
        self.current_target_modules = []
        os.makedirs(workspace_dir, exist_ok=True)
        
    def prune_and_heal(self, dataset, trainer, logger, start_layer=0, num_layers=3, max_steps=1000, search_strategy=None):
        """Iteratively prune layers and apply LoRA healing"""
        current_model = self.model
        steps_per_iter = max_steps // num_layers
        total_steps_used = 0
        
        # Use search strategy if provided
        if search_strategy is not None:
            start_layer = search_strategy.find_start_layer(current_model, self.tokenizer, num_layers)
            print(f"Search strategy found start layer: {start_layer}")
        
        for step in range(num_layers):
            layer_to_remove = start_layer
            layer_for_lora = start_layer - 1 if start_layer > 0 else 0
            
            print(f"\n=== Step {step+1}: Removing layer {layer_to_remove}, LoRA on layer {layer_for_lora} ===")
            
            # Remove layer
            current_model = self._remove_layer(current_model, layer_to_remove)
            layers_remaining = get_model_layers(current_model)
            print(f"Layers remaining: {layers_remaining}")
            
            # Test after pruning
            ppl_after_prune = calculate_perplexity(current_model, self.tokenizer, dataset=dataset.eval_dataset, max_samples=20)
            print(f"Perplexity after pruning: {ppl_after_prune:.3f}")
            
            # Apply LoRA to previous layer
            print(f"Applying LoRA to layer {layer_for_lora}...")
            current_model = self._apply_lora_selective(current_model, layer_for_lora)
            
            # Train only the latest LoRA parameters
            remaining_budget = max_steps - total_steps_used
            current_steps = min(steps_per_iter, remaining_budget)
            print(f"Training LoRA on layer {layer_for_lora} ({current_steps} steps, {total_steps_used}/{max_steps} used)...")
            
            # Freeze all parameters except the latest LoRA
            for name, param in current_model.named_parameters():
                if f"layers.{layer_for_lora}." in name and "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            torch.cuda.empty_cache()
            trainer.model = current_model
            current_model = trainer.train(dataset, max_steps=current_steps)
            total_steps_used += current_steps
            torch.cuda.empty_cache()
            
            # Test after training
            ppl_after_train = calculate_perplexity(current_model, self.tokenizer, dataset=dataset.eval_dataset, max_samples=20)
            print(f"Perplexity after training: {ppl_after_train:.3f}")
            
            # Log step with training info
            logger.log_step({
                "action": "prune", 
                "step": step + 1, 
                "removed_layer": layer_to_remove,
                "lora_layer": layer_for_lora,
                "perplexity": ppl_after_prune,
                "layers_remaining": layers_remaining
            })
            logger.log_step({
                "action": "train", 
                "step": step + 1, 
                "lora_layer": layer_for_lora,
                "perplexity": ppl_after_train,
                "training_steps": current_steps,
                "total_steps_used": total_steps_used,
                "budget_remaining": max_steps - total_steps_used
            })
            
            print(f"Step {step+1} completed! Budget used: {total_steps_used}/{max_steps}")
            
            if total_steps_used >= max_steps:
                print("Training budget exhausted!")
                break
                
        return current_model
    
    def _remove_layer(self, model, layer_idx):
        """Remove specified decoder layer"""
        base_model = self._get_base_model(model)
        with torch.no_grad():
            layers = list(base_model.layers)
            if layer_idx < len(layers):
                layers.pop(layer_idx)
                base_model.layers = nn.ModuleList(layers)
                base_model.config.num_hidden_layers = len(layers)
        return model
    
    def _apply_lora(self, model, layer_idx):
        """Apply LoRA to specified layer"""
        target_modules = self._get_target_modules(model, layer_idx)
        
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        return get_peft_model(model, lora_config)
    
    def _apply_lora_selective(self, model, layer_idx):
        """Apply LoRA to specific layer for selective training"""
        return self._apply_lora(model, layer_idx)
    
    def _get_target_modules(self, model, layer_idx):
        """Get target modules for LoRA based on model architecture"""
        model_type = model.config.model_type.lower()
        
        if "llama" in model_type or "mistral" in model_type:
            return [f"model.layers.{layer_idx}.mlp.gate_proj",
                   f"model.layers.{layer_idx}.mlp.down_proj", 
                   f"model.layers.{layer_idx}.mlp.up_proj"]
        elif "phi" in model_type:
            return [f"model.layers.{layer_idx}.mlp.fc1",
                   f"model.layers.{layer_idx}.mlp.fc2"]
        elif "qwen" in model_type:
            return [f"model.layers.{layer_idx}.mlp.w1",
                   f"model.layers.{layer_idx}.mlp.w2",
                   f"model.layers.{layer_idx}.mlp.c_proj"]
        else:
            return ["gate_proj", "down_proj", "up_proj"]
    
    def _get_base_model(self, model):
        """Get base model from PEFT wrapper if needed"""
        from .utils import get_layers_base
        base = get_layers_base(model)
        if base is None:
            raise RuntimeError(f"Cannot find layers in {model.__class__.__name__}")
        return base

class WindowPruner:
    def __init__(self, model, tokenizer, workspace_dir="./workspace"):
        self.model = model
        self.tokenizer = tokenizer
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
        
    def find_unimportant_window(self, window_size=3):
        """Find least important window of decoder layers"""
        num_layers = get_model_layers(self.model)
        
        if window_size > num_layers:
            print(f"Warning: window_size ({window_size}) > num_layers ({num_layers}), using window_size={num_layers}")
            window_size = num_layers
        
        best_window = None
        best_score = float('inf')
        
        for start_idx in range(num_layers - window_size + 1):
            window = list(range(start_idx, start_idx + window_size))
            score = self._evaluate_window_importance(window)
            
            if score < best_score:
                best_score = score
                best_window = window
        
        # Fallback if no window found
        if best_window is None:
            best_window = list(range(min(3, num_layers)))
            best_score = sum(best_window)
                
        return best_window, best_score
    
    def prune_and_heal(self, dataset, trainer, logger, window_size=3, max_steps=1000, search_strategy=None):
        """Window-based pruning and healing"""
        # Use search strategy if provided, otherwise use default window finding
        if search_strategy is not None:
            start_layer = search_strategy.find_start_layer(self.model, self.tokenizer, window_size)
            window_layers = list(range(start_layer, start_layer + window_size))
            print(f"Search strategy found start layer: {start_layer}")
        else:
            window_layers, _ = self.find_unimportant_window(window_size)
        print(f"Original model layers: {get_model_layers(self.model)}")
        print(f"Removing layers: {window_layers}")
        
        # Remove layers
        base_model = self._get_base_model(self.model)
        with torch.no_grad():
            layers = [layer for i, layer in enumerate(base_model.layers) 
                     if i not in window_layers]
            base_model.layers = nn.ModuleList(layers)
            base_model.config.num_hidden_layers = len(layers)
        
        layers_remaining = get_model_layers(self.model)
        print(f"Remaining layers: {layers_remaining}")
        
        # Test after pruning
        ppl_after_prune = calculate_perplexity(self.model, self.tokenizer, dataset=dataset.eval_dataset, max_samples=20)
        print(f"Perplexity after pruning: {ppl_after_prune:.3f}")
        
        # Apply LoRA to last few MLP layers
        last_layers = min(3, layers_remaining)
        target_modules = []
        for i in range(layers_remaining - last_layers, layers_remaining):
            target_modules.extend([f"model.layers.{i}.mlp.gate_proj", 
                                 f"model.layers.{i}.mlp.down_proj", 
                                 f"model.layers.{i}.mlp.up_proj"])
        
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(self.model, lora_config)
        
        logger.log_step({
            "step": 1,
            "action": "prune",
            "layers_removed": window_layers,
            "layers_remaining": layers_remaining,
            "perplexity": ppl_after_prune
        })
        
        # Train model
        print(f"Fine-tuning with {max_steps} steps...")
        torch.cuda.empty_cache()
        
        trainer.model = model
        model = trainer.train(dataset, max_steps=max_steps)
        torch.cuda.empty_cache()
        
        # Test after training
        final_ppl = calculate_perplexity(model, self.tokenizer, dataset=dataset.eval_dataset, max_samples=20)
        print(f"Final perplexity: {final_ppl:.3f}")
        
        logger.log_step({
            "step": 2,
            "action": "train",
            "perplexity": final_ppl,
            "training_steps": max_steps,
            "total_steps_used": max_steps
        })
        
        return model
    
    def _evaluate_window_importance(self, window):
        """Evaluate importance of layer window (simplified)"""
        # Simplified importance metric - could be enhanced
        return sum(window)  # Placeholder - prefer removing later layers
    
    def _get_mlp_modules(self, layer_idx):
        """Get MLP module names for specific layer"""
        model_type = self.model.config.model_type.lower()
        
        if "llama" in model_type or "mistral" in model_type:
            return [f"model.layers.{layer_idx}.mlp.gate_proj",
                   f"model.layers.{layer_idx}.mlp.down_proj",
                   f"model.layers.{layer_idx}.mlp.up_proj"]
        elif "phi" in model_type:
            return [f"model.layers.{layer_idx}.mlp.fc1",
                   f"model.layers.{layer_idx}.mlp.fc2"]
        elif "qwen" in model_type:
            return [f"model.layers.{layer_idx}.mlp.w1",
                   f"model.layers.{layer_idx}.mlp.w2",
                   f"model.layers.{layer_idx}.mlp.c_proj"]
        else:
            return [f"model.layers.{layer_idx}.mlp.gate_proj",
                   f"model.layers.{layer_idx}.mlp.down_proj",
                   f"model.layers.{layer_idx}.mlp.up_proj"]
    
    def _get_base_model(self, model):
        """Get base model from PEFT wrapper if needed"""
        from .utils import get_layers_base
        base = get_layers_base(model)
        if base is None:
            raise RuntimeError(f"Cannot find layers in {model.__class__.__name__}")
        return base
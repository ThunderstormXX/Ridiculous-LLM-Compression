# pruninghealing/prune.py
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import json
import os
from .utils import get_model_layers, calculate_perplexity

class IterativePruner:
    def __init__(self, model, tokenizer, workspace_dir="./workspace"):
        self.model = model
        self.tokenizer = tokenizer
        self.workspace_dir = workspace_dir
        self.current_target_modules = []
        os.makedirs(workspace_dir, exist_ok=True)
        
    def prune_and_heal(self, start_layer=0, num_layers=5, train_fn=None):
        """Iteratively prune layers and apply LoRA healing"""
        log = []
        current_model = self.model
        
        for step in range(num_layers):
            layer_idx = start_layer + step
            
            # Remove layer
            current_model = self._remove_layer(current_model, layer_idx)
            
            # Apply LoRA to next layer
            current_model = self._apply_lora(current_model, layer_idx)
            
            # Calculate perplexity before training
            pre_ppl = calculate_perplexity(current_model, self.tokenizer)
            
            # Train if function provided
            if train_fn:
                current_model = train_fn(current_model)
                
            # Calculate perplexity after training
            post_ppl = calculate_perplexity(current_model, self.tokenizer)
            
            # Log step
            step_log = {
                "step": step + 1,
                "removed_layer": layer_idx,
                "pre_train_perplexity": pre_ppl,
                "post_train_perplexity": post_ppl,
                "remaining_layers": get_model_layers(current_model)
            }
            log.append(step_log)
            
            # Save checkpoint
            save_path = os.path.join(self.workspace_dir, f"checkpoint_{step+1}")
            current_model.save_pretrained(save_path)
            
        return current_model, log
    
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
        best_window = None
        best_score = float('inf')
        
        for start_idx in range(num_layers - window_size + 1):
            window = list(range(start_idx, start_idx + window_size))
            score = self._evaluate_window_importance(window)
            
            if score < best_score:
                best_score = score
                best_window = window
                
        return best_window, best_score
    
    def prune_window(self, window_layers):
        """Remove window of layers and apply LoRA to last MLP layers"""
        # Remove layers
        base_model = self._get_base_model(self.model)
        with torch.no_grad():
            layers = [layer for i, layer in enumerate(base_model.layers) 
                     if i not in window_layers]
            base_model.layers = nn.ModuleList(layers)
            base_model.config.num_hidden_layers = len(layers)
        
        # Apply QLoRA to last few MLP layers
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Get target modules for last layers
        last_layers = min(3, len(base_model.layers))
        target_modules = []
        for i in range(len(base_model.layers) - last_layers, len(base_model.layers)):
            target_modules.extend(self._get_mlp_modules(i))
        
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        return get_peft_model(self.model, lora_config)
    
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
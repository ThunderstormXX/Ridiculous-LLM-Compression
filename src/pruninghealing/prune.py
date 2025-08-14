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
    """Default strategy for iterative pruning - returns constant layer 24"""
    def find_start_layer(self, model, tokenizer, num_layers):
        return 24

class DefaultWindowStrategy(LayerSearchStrategy):
    """Default strategy for window pruning - returns constant layer 14"""
    def find_start_layer(self, model, tokenizer, num_layers):
        return 14

class IterativePruner:
    def __init__(self, model, tokenizer, workspace_dir="./workspace"):
        self.model = model
        self.tokenizer = tokenizer
        self.workspace_dir = workspace_dir
        self.current_target_modules = []
        self.lora_initialized = False  # ← флаг, что PEFT уже есть
        os.makedirs(workspace_dir, exist_ok=True)
        
    def prune_and_heal(self, dataset, trainer, logger, start_layer=0, num_layers=3, max_steps=1000,
                    search_strategy=None, skip_training=False, skip_perplexity=False, stride=2):
        """Iteratively prune layers and apply LoRA healing.

        By default uses stride=2 which produces the pattern:
        remove start_layer, apply LoRA to start_layer-1,
        remove start_layer-2, apply LoRA to start_layer-3, ...
        """
        current_model = self.model
        steps_per_iter = max_steps // max(1, num_layers)
        total_steps_used = 0

        if search_strategy is not None:
            start_layer = search_strategy.find_start_layer(current_model, self.tokenizer, num_layers)
            print(f"Search strategy found start layer: {start_layer}")

        # Compute removal indices (descending) with given stride (default 2)
        removal_indices = []
        for k in range(num_layers):
            idx = start_layer - k * stride
            if idx < 0:
                print(f"Planned index {idx} < 0: stopping early (will remove {len(removal_indices)} layers).")
                break
            removal_indices.append(idx)

        if len(removal_indices) == 0:
            print("No layers to remove (start_layer too small).")
            return current_model

        print(f"Planned removal sequence (descending): {removal_indices}")

        for step, layer_to_remove in enumerate(removal_indices):
            # compute layer for LoRA (previous layer)
            layer_for_lora = layer_to_remove - 1 if layer_to_remove > 0 else 0

            print(f"\n=== Step {step+1}: Removing layer {layer_to_remove}, LoRA on layer {layer_for_lora} ===")

            # Remove layer (safe: we remove from high indices to low, so lower indices stay stable)
            current_model = self._remove_layer(current_model, layer_to_remove)
            num_layers_remaining = get_model_layers(current_model)
            print(f"Layers remaining: {num_layers_remaining}")

            # Test after pruning
            if not skip_perplexity and dataset and dataset.eval_dataset:
                ppl_after_prune = calculate_perplexity(current_model, self.tokenizer, dataset=dataset.eval_dataset)
                print(f"Perplexity after pruning: {ppl_after_prune:.3f}")
            else:
                ppl_after_prune = 0.0
                print("Skipping perplexity calculation (debug mode)")

            # Apply LoRA to previous layer (always, if exists)
            if layer_for_lora >= 0:
                print(f"Applying LoRA to layer {layer_for_lora}...")
                current_model = self._apply_lora_selective(current_model, layer_for_lora)
            else:
                print("No previous layer to apply LoRA to (index < 0).")

            if not skip_training:
                # Train only the latest LoRA parameters
                remaining_budget = max_steps - total_steps_used
                current_steps = min(steps_per_iter, remaining_budget)
                print(f"Training LoRA on layer {layer_for_lora} ({current_steps} steps, {total_steps_used}/{max_steps} used)...")

                # First freeze everything
                for _, param in current_model.named_parameters():
                    param.requires_grad = False

                # Then enable training for adapter-related parameters only.
                # We rely on adapter name pattern used in _apply_lora_selective: "lora_layer_{layer_idx}"
                adapter_name = f"lora_layer_{layer_for_lora}"
                enabled_any = False
                for name, param in current_model.named_parameters():
                    # Enable params that belong to this adapter or contain both layer path and 'lora'
                    if adapter_name in name or (f"layers.{layer_for_lora}." in name and "lora" in name):
                        param.requires_grad = True
                        enabled_any = True

                if not enabled_any:
                    # Fallback: enable any parameter with 'lora' in name (defensive)
                    for name, param in current_model.named_parameters():
                        if "lora" in name:
                            param.requires_grad = True

                torch.cuda.empty_cache()
                trainer.model = current_model
                current_model = trainer.train(dataset, max_steps=current_steps)
                total_steps_used += current_steps
                torch.cuda.empty_cache()

                # Test after training
                if not skip_perplexity and dataset and dataset.eval_dataset:
                    ppl_after_train = calculate_perplexity(current_model, self.tokenizer, dataset=dataset.eval_dataset)
                    print(f"Perplexity after training: {ppl_after_train:.3f}")
                else:
                    ppl_after_train = 0.0
                    print("Skipping perplexity calculation (debug mode)")
            else:
                print("Skipping training (debug mode)")
                ppl_after_train = ppl_after_prune

            # Log step with training info
            logger.log_step({
                "action": "prune",
                "step": step + 1,
                "removed_layer": layer_to_remove,
                "lora_layer": layer_for_lora,
                "perplexity": ppl_after_prune,
                "layers_remaining": num_layers_remaining
            })
            if not skip_training:
                logger.log_step({
                    "action": "train",
                    "step": step + 1,
                    "lora_layer": layer_for_lora,
                    "perplexity": ppl_after_train,
                    "training_steps": current_steps,
                    "total_steps_used": total_steps_used,
                    "budget_remaining": max_steps - total_steps_used
                })
            else:
                logger.log_step({
                    "action": "skip_train",
                    "step": step + 1,
                    "lora_layer": layer_for_lora,
                    "perplexity": ppl_after_train
                })

            if not skip_training:
                print(f"Step {step+1} completed! Budget used: {total_steps_used}/{max_steps}")
                if total_steps_used >= max_steps:
                    print("Training budget exhausted!")
                    break
            else:
                print(f"Step {step+1} completed! (debug mode - no training)")
        
        
        print(current_model)
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
    
    def _apply_lora_selective(self, model, layer_idx):
        target_modules = self._get_target_modules(model, layer_idx)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        from peft import PeftModel

        if not self.lora_initialized:
            model = get_peft_model(model, lora_config)
            self.lora_initialized = True
        else:
            if not isinstance(model, PeftModel):
                raise RuntimeError("Model is not PEFT-wrapped but lora_initialized=True")
            model.add_adapter(f"lora_layer_{layer_idx}", lora_config)
            model.set_adapter(f"lora_layer_{layer_idx}")

        return model
    def _get_base_model(self, model):
        """Get base model from PEFT wrapper if needed"""
        from .utils import get_layers_base
        from peft import PeftModel

        while isinstance(model, PeftModel):
            model = model.get_base_model()
        base = get_layers_base(model)
        if base is None:
            raise RuntimeError(f"Cannot find layers in {model.__class__.__name__}")
        return base






#     def _get_base_model(self, model):
#         # Возвращает базовую модель с .layers, например, model.model или model.base_model
#         if hasattr(model, "model"):
#             return model.model
#         elif hasattr(model, "base_model"):
#             return model.base_model
#         else:
#             return model  # на крайний случай


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
    
    def prune_and_heal(self, dataset, trainer, logger, window_size=3, max_steps=1000, search_strategy=None, skip_training=False, skip_perplexity=False):
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
        
        num_layers_remaining = get_model_layers(self.model)
        print(f"Remaining layers: {num_layers_remaining}")
        
        # Test after pruning
        if not skip_perplexity and dataset and dataset.eval_dataset:
            ppl_after_prune = calculate_perplexity(self.model, self.tokenizer, dataset=dataset.eval_dataset)
            print(f"Perplexity after pruning: {ppl_after_prune:.3f}")
        else:
            ppl_after_prune = 0.0
            print("Skipping perplexity calculation (debug mode)")
        
        logger.log_step({
            "step": 1,
            "action": "prune",
            "layers_removed": window_layers,
            "layers_remaining": num_layers_remaining,
            "perplexity": ppl_after_prune
        })
        
        # Apply LoRA to all remaining MLP layers (always)
        target_modules = []
        for i in range(num_layers_remaining):
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
        
        if not skip_training:
            # Train model
            print(f"Fine-tuning with {max_steps} steps...")
            torch.cuda.empty_cache()
            
            trainer.model = model
            model = trainer.train(dataset, max_steps=max_steps)
            torch.cuda.empty_cache()
            
            # Test after training
            if not skip_perplexity and dataset and dataset.eval_dataset:
                final_ppl = calculate_perplexity(model, self.tokenizer, dataset=dataset.eval_dataset)
                print(f"Final perplexity: {final_ppl:.3f}")
            else:
                final_ppl = 0.0
                print("Skipping perplexity calculation (debug mode)")
            
            logger.log_step({
                "step": 2,
                "action": "train",
                "perplexity": final_ppl,
                "training_steps": max_steps,
                "total_steps_used": max_steps
            })
        else:
            print("Skipping training (debug mode)")
            final_ppl = ppl_after_prune
            
            logger.log_step({
                "step": 2,
                "action": "skip_train",
                "perplexity": final_ppl
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
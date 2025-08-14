# pruninghealing/importance_strategies.py
import numpy as np
from .prune import LayerSearchStrategy

class ImportanceBasedIterativeStrategy(LayerSearchStrategy):
    """Strategy for iterative pruning based on layer importance"""
    def __init__(self, importances):
        self.importances = importances
        
    def find_start_layer(self, model, tokenizer, num_layers):
        print("Layer importances (sorted by importance):")
        for layer_idx, importance in self.importances:
            print(f"  Layer {layer_idx}: {importance:.6f}")
        
        imp_dict = {layer_idx: importance for layer_idx, importance in self.importances}
        best_start = None
        best_sum = float('inf')
        
        max_layer = max(imp_dict.keys())
        for start in range(num_layers * 2, max_layer + 1):
            removal_sum = 0
            valid = True
            
            for k in range(num_layers):
                layer_idx = start - k * 2
                if layer_idx < 0 or layer_idx not in imp_dict:
                    valid = False
                    break
                removal_sum += imp_dict[layer_idx]
            
            if valid and removal_sum < best_sum:
                best_sum = removal_sum
                best_start = start
        
        if best_start is None:
            best_start = 24
            best_sum = 0
        
        removal_layers = [best_start - k * 2 for k in range(num_layers) if best_start - k * 2 >= 0]
        print(f"Selected start layer: {best_start}")
        print(f"Will remove layers: {removal_layers}")
        print(f"Total importance sum: {best_sum:.6f}")
        
        all_importances = [imp for _, imp in self.importances]
        mean_importance = np.mean(all_importances)
        var_importance = np.var(all_importances)
        print(f"Mean importance: {mean_importance:.6f}")
        print(f"Importance variance: {var_importance:.6f}")
        
        return best_start

class ImportanceBasedWindowStrategy(LayerSearchStrategy):
    """Strategy for window pruning based on layer importance"""
    def __init__(self, importances):
        self.importances = importances
        
    def find_start_layer(self, model, tokenizer, num_layers):
        print("Layer importances (sorted by importance):")
        for layer_idx, importance in self.importances:
            print(f"  Layer {layer_idx}: {importance:.6f}")
        
        imp_dict = {layer_idx: importance for layer_idx, importance in self.importances}
        best_start = None
        best_sum = float('inf')
        
        max_layer = max(imp_dict.keys())
        for start in range(max_layer - num_layers + 2):
            window_sum = 0
            valid = True
            
            for i in range(num_layers):
                layer_idx = start + i
                if layer_idx not in imp_dict:
                    valid = False
                    break
                window_sum += imp_dict[layer_idx]
            
            if valid and window_sum < best_sum:
                best_sum = window_sum
                best_start = start
        
        if best_start is None:
            best_start = 14
            best_sum = 0
        
        window_layers = list(range(best_start, best_start + num_layers))
        print(f"Selected start layer: {best_start}")
        print(f"Will remove window: {window_layers}")
        print(f"Window importance sum: {best_sum:.6f}")
        
        all_importances = [imp for _, imp in self.importances]
        mean_importance = np.mean(all_importances)
        var_importance = np.var(all_importances)
        print(f"Mean importance: {mean_importance:.6f}")
        print(f"Importance variance: {var_importance:.6f}")
        
        return best_start
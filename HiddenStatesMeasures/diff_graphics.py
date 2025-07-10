import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def visualize_heterogeneous_metrics(hidden_states, max_prompts=10, font_size=22):
    """
    Visualize metrics for hidden states with varying shapes
    
    Args:
        hidden_states: Dictionary {
            'component_name': {
                layer_idx: [sample1_state, sample2_state, ...],  # States may have different shapes
                ...
            },
            ...
        }
    """
    components = list(hidden_states.keys())
    layers = sorted(hidden_states[components[0]].keys())
    num_layers = len(layers)
    
    # Store metrics per sample then average
    sample_metrics = {
        'mean_diff': {comp: [[] for _ in range(num_layers-1)] for comp in components},
        'mean_abs_diff': {comp: [[] for _ in range(num_layers-1)] for comp in components},
        'std_diff': {comp: [[] for _ in range(num_layers-1)] for comp in components},
        'cosine_sim': {comp: [[] for _ in range(num_layers-1)] for comp in components}
    }
    
    # Calculate metrics for each sample individually
    for comp in components:
        for layer_idx in range(num_layers - 1):
            current_layer = hidden_states[comp][layer_idx]
            next_layer = hidden_states[comp][layer_idx + 1]
            ixx_ = 0
            for current_state, next_state in zip(current_layer, next_layer):
                ixx_ += 1
                # Convert to numpy if they're torch tensors
                current = current_state.detach().cpu().to(torch.float32).numpy()
                next_ = next_state.detach().cpu().to(torch.float32).numpy()
                # current = current_state.numpy() if hasattr(current_state, 'numpy') else current_state
                # next_ = next_state.numpy() if hasattr(next_state, 'numpy') else next_state
                
                # Ensure 2D (seq_len, hidden_dim)
                current = current.reshape(-1, current.shape[-1])
                next_ = next_.reshape(-1, next_.shape[-1])
                
                # Pad or truncate to match sequence lengths
                min_seq_len = min(current.shape[0], next_.shape[0])
                current = current[:min_seq_len]
                next_ = next_[:min_seq_len]
                
                # Compute metrics for this sample
                diff = next_ - current
                sample_metrics['mean_diff'][comp][layer_idx].append(np.mean(diff))
                sample_metrics['mean_abs_diff'][comp][layer_idx].append(np.mean(np.abs(diff)))
                sample_metrics['std_diff'][comp][layer_idx].append(np.std(diff))
                
                # Cosine similarity (pairwise)
                cos_sim = np.mean([cosine_similarity([c], [n])[0][0] 
                                 for c, n in zip(current, next_)])
                sample_metrics['cosine_sim'][comp][layer_idx].append(cos_sim)
                if ixx_ == max_prompts:
                    break
    
    # Average across samples
    metrics = {
        'mean_diff': {comp: [] for comp in components},
        'mean_abs_diff': {comp: [] for comp in components},
        'std_diff': {comp: [] for comp in components},
        'cosine_sim': {comp: [] for comp in components}
    }
    
    for metric in metrics.keys():
        for comp in components:
            metrics[metric][comp] = [
                np.mean(layer_metrics) 
                for layer_metrics in sample_metrics[metric][comp]
            ]
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    plt.rcParams.update({'font.size': font_size})
    
    for ax, (metric_name, comp_values) in zip(axes, metrics.items()):
        for comp in components:
            ax.plot(range(num_layers - 1), comp_values[comp], 
                   label=comp, marker='o')
        
        ax.set_xlabel('Layer Number (n → n+1)')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} Across Layers')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return metrics

def visualize_heterogeneous_metrics2(hidden_states, max_prompts=10, font_size=22):
    """
    Visualize metrics for hidden states with varying shapes
    
    Args:
        hidden_states: Dictionary {
            'component_name': {
                layer_idx: [sample1_state, sample2_state, ...],  # States may have different shapes
                ...
            },
            ...
        }
    """
    components = list(hidden_states.keys())
    layers = sorted(hidden_states[components[0]].keys())
    num_layers = len(layers)
    
    # Store metrics per sample then average
    sample_metrics = {
        'mean_diff': {comp: [[] for _ in range(num_layers-1)] for comp in components},
        'mean_abs_diff': {comp: [[] for _ in range(num_layers-1)] for comp in components},
        'std_diff': {comp: [[] for _ in range(num_layers-1)] for comp in components},
        'cosine_sim': {comp: [[] for _ in range(num_layers-1)] for comp in components}
    }
    
    # Calculate metrics for each sample individually
    for layer_idx in range(num_layers - 1):
        for c_ix, comp in enumerate(components):
            
            current_layer = hidden_states[components[(c_ix - 1) % len(components)]][layer_idx]
            next_layer = hidden_states[comp][layer_idx]
            ixx_ = 0
            for current_state, next_state in zip(current_layer, next_layer):
                ixx_ += 1
                # Convert to numpy if they're torch tensors
                current = current_state.detach().cpu().to(torch.float32).numpy()
                next_ = next_state.detach().cpu().to(torch.float32).numpy()
                # current = current_state.numpy() if hasattr(current_state, 'numpy') else current_state
                # next_ = next_state.numpy() if hasattr(next_state, 'numpy') else next_state
                
                # Ensure 2D (seq_len, hidden_dim)
                current = current.reshape(-1, current.shape[-1])
                next_ = next_.reshape(-1, next_.shape[-1])
                
                # Pad or truncate to match sequence lengths
                min_seq_len = min(current.shape[0], next_.shape[0])
                current = current[:min_seq_len]
                next_ = next_[:min_seq_len]
                
                # Compute metrics for this sample
                diff = next_ - current
                sample_metrics['mean_diff'][comp][layer_idx].append(np.mean(diff))
                sample_metrics['mean_abs_diff'][comp][layer_idx].append(np.mean(np.abs(diff)))
                sample_metrics['std_diff'][comp][layer_idx].append(np.std(diff))
                
                # Cosine similarity (pairwise)
                cos_sim = np.mean([cosine_similarity([c], [n])[0][0] 
                                 for c, n in zip(current, next_)])
                sample_metrics['cosine_sim'][comp][layer_idx].append(cos_sim)
                if ixx_ == max_prompts:
                    break
    
    # Average across samples
    metrics = {
        'mean_diff': {comp: [] for comp in components},
        'mean_abs_diff': {comp: [] for comp in components},
        'std_diff': {comp: [] for comp in components},
        'cosine_sim': {comp: [] for comp in components}
    }
    
    for metric in metrics.keys():
        for comp in components:
            metrics[metric][comp] = [
                np.mean(layer_metrics) 
                for layer_metrics in sample_metrics[metric][comp]
            ]
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    plt.rcParams.update({'font.size': font_size})
    
    for ax, (metric_name, comp_values) in zip(axes, metrics.items()):
        for c_ix, comp in enumerate(components):
            ax.plot(range(num_layers - 1), comp_values[comp], 
                   label=components[(c_ix - 1) % len(components)] + ' → ' + comp, marker='o')
        print(metrics['std_diff'])
        ax.set_xlabel('Element Transition (n)')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} Across Layers')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return metrics

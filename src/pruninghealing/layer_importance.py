# pruninghealing/layer_importance.py
import json
import numpy as np

def calc_mean_abs_diff(a0, a1):
    """Calculate mean absolute difference between two arrays"""
    delta = np.abs(a0 - a1)
    delta[delta == np.inf] = np.nan
    return np.nanmean(delta)

def cosine_vectors(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    mask = ~np.isinf(vec1) & ~np.isinf(vec2)
    vec1_filtered = vec1[mask]
    vec2_filtered = vec2[mask]
    dot_product = np.dot(vec1_filtered, vec2_filtered)
    norm_vec1 = np.linalg.norm(vec1_filtered)
    norm_vec2 = np.linalg.norm(vec2_filtered)
    if norm_vec1 == 0 or norm_vec2 == 0 or not len(vec1_filtered):
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def calc_mean_cosine(a0, a1, shape):
    """Calculate mean cosine similarity between reshaped arrays"""
    a0[np.isnan(a0)] = np.inf
    a1[np.isnan(a1)] = np.inf
    if max(np.max(np.abs(a0)), np.max(np.abs(a1))) == np.inf:
        vals = []
        for i in range(len(a0)):
            vals.append(cosine_vectors(a0[i], a1[i]))
        return np.mean(vals)
    A = a0.reshape(shape)
    B = a1.reshape(shape)
    return np.mean(np.sum(A * B, axis=1) 
                   / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)))

def compute_layer_importances(hidden_path, layer_type='mlp'):
    """Compute layer importances from hidden states"""
    with open(hidden_path, 'r') as f:
        hiddens = json.load(f)

    n_layers = len(hiddens[layer_type])
    n_questions = len(hiddens[layer_type]['0'])
    
    diffs = {i: {j: [] for j in range(n_layers-1)} for i in ['abs', 'std', 'cos']}
    
    for n in range(n_layers-1):
        for q in range(n_questions):
            prev = np.array(hiddens[layer_type][str(n)]['hidden_states'][q])
            cur = np.array(hiddens[layer_type][str(n+1)]['hidden_states'][q])
            shape = hiddens[layer_type][str(n)]['shapes'][q]
            diffs['abs'][n].append(calc_mean_abs_diff(cur, prev))
            diffs['std'][n].append(calc_mean_abs_diff(cur, prev))
            diffs['cos'][n].append(1 - calc_mean_cosine(cur, prev, shape))

    # Convert to importances (sorted by importance)
    scores = [[k, np.mean(i)] for k, i in diffs['cos'].items()]
    return sorted(scores, key=lambda x: x[1])
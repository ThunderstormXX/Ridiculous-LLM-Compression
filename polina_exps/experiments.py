# %%
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

device = 'cuda:0'

# 1. Инициализация модели и токенизатора
model = LlamaForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")

# 2. Подготовка 20 образцов MMLU (dev)
ds = load_dataset("cais/mmlu", "all",split="dev")
smpl = ds.shuffle(seed=42).select(range(20))
texts = [
    item["question"] + " Варианты: " + " ".join(item["choices"])
    for item in smpl
]
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
)


# %%
inputs['input_ids'] = inputs['input_ids'].to(device)
inputs['attention_mask'] = inputs['attention_mask'].to(device)

# %%
# Собираем скрытые состояния до каждого слоя
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)
hidden_states = out.hidden_states  # список длины num_layers+1

# %%
model


# %%
import inspect

print(inspect.getsource(type(model.model.layers[0])))

# %%
inputs = {k: v.to(model.device) for k, v in inputs.items()}
inputs['attention_mask'] = torch.tril(torch.ones(100, 100, dtype=bool, device=device))

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // num_heads
layer_idx = num_layers // 2  # средний слой

# %%
position_ids = torch.arange(
    0, 100, dtype = torch.long, device = device
)

position_embeddings = model.model.rotary_emb(inputs['input_ids'], position_ids[None, :])

# %%
model.model.layers[0]

# %%
from copy import deepcopy

# 3. Implementing the logging function for layer inspection
forward_func = deepcopy(model.model.layers[11].forward)  # Copy original forward method of layer 11
logger = []

def new_forward(*args, **kwargs):
    # Log inputs
    layer_inputs = {
        'hidden_states': args[0] if len(args) > 0 else kwargs.get('hidden_states'),
        'attention_mask': kwargs.get('attention_mask'),
        'position_ids': kwargs.get('position_ids'),
        'past_key_value': kwargs.get('past_key_value'),
        'output_attentions': kwargs.get('output_attentions'),
        'use_cache': kwargs.get('use_cache')
    }
    logger.append({'inputs': layer_inputs})
    
    # Call original forward
    outputs = forward_func(*args, **kwargs)
    
    # Log outputs
    layer_outputs = {
        'hidden_states': outputs[0],
        'attentions': outputs[1] if len(outputs) > 1 else None,
        'past_key_value': outputs[2] if len(outputs) > 2 else None
    }
    logger[-1]['outputs'] = layer_outputs
    
    return outputs

# Replace the forward method of layer 11 with our logging version
model.model.layers[11].forward = new_forward

# 4. Process prompts and collect logs
prompts = [
    "Explain the theory of relativity",
    "What is the capital of France?",
    "How to bake a chocolate cake?"
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # You can now analyze the logger content
    print(f"Processed prompt: {prompt}")
    print(f"Number of logged operations: {len(logger)}")
    
    # Example: Print input/output shapes for first logged operation
    if logger:
        first_log = logger[0]
        print("Input hidden states shape:", first_log['inputs']['hidden_states'].shape)
        print("Output hidden states shape:", first_log['outputs']['hidden_states'].shape)
    
    # Clear logger for next prompt
    logger = []

# %%


# %%
log = logger[0]
log.keys()

# %%
log['input'], log['att_nmaks']

# %%
x = hidden_states[0]
x.shape

# %%
model.model.layers[10].forward(x)

# %%
for layer in model.model.layers
    arr.append(layer.self_attn.q k v )


q , k, v --> listq listk listv 

arrq, arrk,arrv 

arrq.shape = (num_prompts, num_heads, H,W)

arrq



# %%
head_dim

# %%
def extract_head_repr_forward (layer_idx, head_idx, position_embeddings, hidden_states, inputs, model):
    """
    Извлекаем представление заданной головы через forward self-attn одного слоя.
    
    Args:
        layer_idx: Индекс слоя
        head_idx: Индекс головы
        position_embeddings: Позиционные эмбеддинги
        hidden_states: Скрытые состояния модели
        inputs: Входные данные (содержит attention_mask)
        model: Модель трансформера
    
    Returns:
        Векторное представление головы [hidden_size]
    """
    # Получаем модуль слоя
    layer = model.model.layers[layer_idx]
    inp = hidden_states[layer_idx]  # [batch, seq, hidden_size]
    
    # Сохраняем оригинальные веса
    orig_q_weight = layer.self_attn.q_proj.weight.data.clone()
    orig_k_weight = layer.self_attn.k_proj.weight.data.clone()
    orig_v_weight = layer.self_attn.v_proj.weight.data.clone()

    try:
        with torch.no_grad():
            num_heads = 24
            head_dim = 128

            # Создаем маску для выбранной головы
            mask = torch.zeros(num_heads, device= model.device)
            mask[head_idx] = 1.0  

            # Применяем маски к проекционным матрицам
            q_mask = mask.repeat_interleave(head_dim)
            print(mask.shape, q_mask.shape, layer.self_attn.q_proj.weight.data.shape)
            layer.self_attn.q_proj.weight.data *= q_mask

            k_head_dim = layer.self_attn.k_proj.weight.shape[0] // num_heads
            v_head_dim = layer.self_attn.v_proj.weight.shape[0] // num_heads
            
            k_mask = mask.repeat_interleave(k_head_dim)
            v_mask = mask.repeat_interleave(v_head_dim)

            layer.self_attn.k_proj.weight.data *= k_mask
            layer.self_attn.v_proj.weight.data *= v_mask

            # Forward pass через self-attention
            attn_out = layer.self_attn(
                inp,
                attention_mask=inputs.get("attention_mask"), 
                position_embeddings=position_embeddings, 
            )[0]

            # Усредняем по батчу и последовательности
            rep = attn_out.mean(dim=(0,1))  # [hidden_size]
            
    finally:
        # Восстанавливаем оригинальные веса
        layer.self_attn.q_proj.weight.data.copy_(orig_q_weight)
        layer.self_attn.k_proj.weight.data.copy_(orig_k_weight)
        layer.self_attn.v_proj.weight.data.copy_(orig_v_weight)

    return rep.detach().cpu().numpy()

# %%
import inspect

print(inspect.getsource(type(model.model.layers[0].self_attn)))

# %%
import tqdm
layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
layer_reprs = {}

for li in tqdm.tqdm(layers, desc="Analyzing layers"):
    try:
        # Извлекаем представления всех голов
        reps = np.stack([
            extract_head_repr_forward 
            (layer_idx, 
             head_idx, 
             position_embeddings, 
             hidden_states, 
             inputs, 
             model)
            for head_idx in range(num_heads)
        ], axis=0)
        
        layer_reprs[li] = reps
        print(f"Layer {li}: extracted {reps.shape[0]} heads with dim {reps.shape[1]}")
        
    except Exception as e:
        raise e
        print(f"Error in layer {li}: {str(e)}")
        continue

# %%


# %%


# %%
norms = np.linalg.norm(reprs, axis=1, keepdims=True)

# %%
# Косинус и L2

norms = np.linalg.norm(reprs, axis=1, keepdims=True)
cos_sim = (reprs @ reprs.T) / (norms @ norms.T)

l2_dist = squareform(pdist(reprs, metric='euclidean'))

# Find similar pairs by threshold
threshold = 0.95
pairs = []
for i in range(num_heads):
    for j in range(i+1, num_heads):
        if cos_sim[i,j] >= threshold:
            pairs.append((i, j, float(cos_sim[i,j]), float(l2_dist[i,j])))

pairs_sorted = sorted(pairs, key=lambda x: -x[2])
print(f"Found {len(pairs_sorted)} pairs with cos >= {threshold}")

# %%
# Функция визуализации
def visualize_layer_attention(layer_idx):
    torch.cuda.empty_cache()
    with torch.no_grad():
        out_att = model(**inputs, output_attentions=True).attentions[layer_idx]
    avg_att = out_att.mean(dim=0).cpu().numpy()  # [heads, seq, seq]

    buttons = []
    for h in range(avg_att.shape[0]):
        btn = widgets.Button(description=f"Head {h}", layout=widgets.Layout(width='80px'))
        def on_click_factory(mat, li, idx):
            def on_click(_):
                plt.figure(figsize=(6,5))
                sns.heatmap(mat[idx], cmap='magma')
                plt.title(f"Layer {li}, Head {idx}")
                plt.show()
            return on_click
        btn.on_click(on_click_factory(avg_att, layer_idx, h))
        buttons.append(btn)

    display(widgets.VBox([widgets.Label(f"Layer {layer_idx}"), widgets.HBox(buttons)]))

# Вызов визуализации
visualize_layer_attention(layer_idx)



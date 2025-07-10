# Ridiculous LLM Compression
“Ridiculous LLM Compression Techniques” project by Ignashin Igor, Kiselyov Ivan, Leontyeva Polina, Murzina Anastasiya, and mentor Bulatov Aydar (AIRI Summer School 2025).

This repository accompanies the report “Ridiculous LLM Compression Techniques” and explores practical ways to shrink LLaMA models while keeping performance high. The goal is to reduce GPU memory usage so that large language models can be deployed on modest hardware.

## Key Experiments
Attention Head Similarity – measuring cosine similarity between attention heads to identify redundant heads that can be pruned.

Double‑Layer Removal – systematically removing pairs of layers to reveal which layers are essential and which are not.

Layer Interchangeability – replacing the weights of one layer with those of another to test functional overlap.

Centered Kernel Alignment (CKA) – comparing layer activations to discover layers with similar representations.

Layer Interpolation – creating blended layers by averaging parameters of two layers and observing the effect on perplexity.

Iterative LoRA Healing – gradually pruning layers while adding LoRA adapters to the remaining layers to recover quality.

Full details and results are provided in Report.pdf.

## Repository Layout
HiddenStatesMeasures/ – scripts and notebooks for analyzing hidden state dynamics, such as diff_graphics.py and inference_tiny_llama_reduce.ipynb.

PruningHealing/ – LoRA‑based pruning workflow, containing main.py, various notebooks, and experiment logs.

hvostchedUser/ – utilities for layer replacement analysis and perplexity evaluation.

finding_similar_heads.ipynb – notebook for the attention head similarity study.

## Running the Experiments
Environment Setup
```
conda create -n llama-compress python=3.10
pip install -r PruningHealing/requirements.txt
```
Iterative Pruning with LoRA
```
python PruningHealing/main.py
```
This script loads the specified LLaMA model and performs the iterative pruning + LoRA healing procedure described in the report.

Other scripts and notebooks follow assumptions: a working Python 3.10 environment and the dependencies listed in PruningHealing/requirements.txt.

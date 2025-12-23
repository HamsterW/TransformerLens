import os
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch as t
import functools
import sys
from pathlib import Path
from typing import Callable

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint
from transformers import AutoModel, AutoTokenizer

import plotly.express as px
from transformer_lens.utils import to_numpy

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# print(gpt2_small.cfg)

tl_model = HookedTransformer.from_pretrained(
    'GSAI-ML/LLaDA-8B-Base', 
    trust_remote_code=True,
    n_devices=1,
    dtype="float16",
)

print(tl_model.cfg)

# # 1. Run a tiny test
# prompt = "The capital of France is"
# tokens = tl_model.to_tokens(prompt)

# # 2. Capture the Attention Pattern
# # We filter to only save patterns to avoid OOM
# logits, cache = tl_model.run_with_cache(tokens, names_filter=lambda n: "pattern" in n)

# # 3. Visualize Layer 15, Head 0 (A middle layer)
# # Shape is [batch, head, query_pos, key_pos]
# # We squeeze to get [query_pos, key_pos]
# attn_matrix = cache["pattern", 15][0, 0]

# # 4. Plot
# # We mask out the lower triangle (future) for clarity, though LLaDA is bidirectional
# output_file = "attention_pattern.html"

# px.imshow(
#     to_numpy(attn_matrix),
#     title="Attention Diagnostic: Layer 15 Head 0",
#     labels={"x": "Key (Source)", "y": "Query (Destination)"},
#     x=tl_model.to_str_tokens(tokens),
#     y=tl_model.to_str_tokens(tokens)
# ).write_html(output_file)


def generate_llada(model, prompt, steps=64, gen_len=32):
    """
    Generates text using LLaDA's Masked Diffusion process.
    """
    # 1. Setup Constants
    # LLaDA uses a specific token ID for [MASK]. 
    # Check your config, but usually it is 126336 for LLaDA-8B
    MASK_ID = 126336 
    
    # 2. Prepare Input
    # [Prompt Tokens] + [MASK Tokens]
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True).squeeze(0)
    num_prompt = len(prompt_tokens)
    
    # Create the canvas: Prompt followed by empty masks
    mask_canvas = t.full((gen_len,), MASK_ID, dtype=t.long, device=model.cfg.device)
    input_ids = t.cat([prompt_tokens, mask_canvas]).unsqueeze(0) # Shape: [1, Seq_Len]
    
    print(f"Starting Generation: '{prompt}' + {gen_len} masks")

    # 3. The Diffusion Loop (The "Refinement" Process)
    # We iterate from T (high noise) down to 0 (no noise)
    for step in range(steps):
        # Calculate how many tokens we are allowed to keep this round
        # Linear schedule: At step 0, we mask 100%. At step 64, we mask 0%.
        progress = (step + 1) / steps
        tokens_to_keep_ratio = progress
        
        # A. Run the model (Forward Pass)
        # We use run_with_cache if you want to inspect, otherwise just model()
        logits = model(input_ids) 
        
        # Focus only on the generated part (ignore the prompt part)
        gen_logits = logits[0, num_prompt:] 
        
        # B. Get Predictions and Confidence Scores
        probs = F.softmax(gen_logits, dim=-1)
        confidence, predicted_ids = t.max(probs, dim=-1)
        predicted_ids = t.multinomial(probs, num_samples=1).squeeze()
        
        # C. The "Re-masking" Strategy
        # We want to keep the top X% most confident tokens
        num_to_keep = int(gen_len * tokens_to_keep_ratio)
        
        # Sort by confidence to find the best ones
        # topk returns the values and the INDICES in the sequence
        top_conf, top_indices = t.topk(confidence, k=num_to_keep)
        
        # Create the new input for the next step
        # Start with all masks again
        new_gen_ids = t.full((gen_len,), MASK_ID, dtype=t.long, device=model.cfg.device)
        
        # Fill in ONLY the high-confidence tokens we decided to keep
        new_gen_ids[top_indices] = predicted_ids[top_indices]
        
        # Update the main input array
        input_ids[0, num_prompt:] = new_gen_ids
        
        # Optional: Print intermediate result to see the "thought process"
        if step % 10 == 0:
            current_text = model.tokenizer.decode(new_gen_ids)
            print(f"Step {step}/{steps}: {current_text}")

    return model.to_string(input_ids[0])

# --- Usage ---
output = generate_llada(tl_model, "The capital of France is", steps=50, gen_len=10)
print("\nFinal Output:", output)
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

tl_model = HookedTransformer.from_pretrained(
    'GSAI-ML/LLaDA-8B-Base', 
    trust_remote_code=True,
    n_devices=1,
    dtype="float16",
)

print(tl_model)
print(tl_model.cfg)

# def compare_weights(tl_model, hf_model_name="GSAI-ML/LLaDA-8B-Base"):
#     print(f"Loading reference model: {hf_model_name}...")
#     # Load HF model to CPU to avoid OOM
#     hf_model = AutoModelForCausalLM.from_pretrained(
#         hf_model_name, 
#         trust_remote_code=True, 
#         torch_dtype=torch.float16,
#         device_map="cpu"
#     )
    
#     base = hf_model.model.transformer
#     cfg = tl_model.cfg
#     print("\n--- Starting Weight Audit ---")

#     # 1. Compare Embeddings
#     print("Checking Embeddings...")
#     hf_embed = base.wte.weight
#     tl_embed = tl_model.W_E.cpu() # Move to CPU for comparison
    
#     diff = (hf_embed - tl_embed).abs().max().item()
#     if diff < 1e-4:
#         print(f"✅ Embeddings match (Max Diff: {diff:.6f})")
#     else:
#         print(f"❌ Embeddings MISMATCH (Max Diff: {diff:.6f})")

#     # 2. Check a Middle Layer (e.g., Layer 15) to verify blocks
#     layer_idx = 15
#     print(f"\nChecking Layer {layer_idx}...")
    
#     hf_block = base.blocks[layer_idx]
#     tl_block = tl_model.blocks[layer_idx]

#     # --- Helper to reshape HF attention weights to TL shape ---
#     # HF: [n_heads * d_head, d_model] -> TL: [n_heads, d_model, d_head]
#     def to_tl_attn(hf_weight):
#         # 1. Reshape to [n_heads, d_head, d_model]
#         w = einops.rearrange(hf_weight, "(n h) m -> n h m", n=cfg.n_heads, h=cfg.d_head)
#         # 2. Transpose last two dims to match TL [n_heads, d_model, d_head]
#         return w.transpose(-1, -2)

#     # --- Compare Attention (W_Q) ---
#     # Note: If you unpermuted RoPE, you might need to apply that here too to match!
#     # For a raw check, we assume standard loading.
#     hf_q = to_tl_attn(hf_block.q_proj.weight)
#     tl_q = tl_block.attn.W_Q.cpu()
    
#     diff_q = (hf_q - tl_q).abs().max().item()
#     if diff_q < 1e-4:
#         print(f"✅ Attention W_Q matches (Max Diff: {diff_q:.6f})")
#     else:
#         print(f"❌ Attention W_Q MISMATCH (Max Diff: {diff_q:.6f})")
#         print(f"   - HF Shape: {hf_q.shape}")
#         print(f"   - TL Shape: {tl_q.shape}")

#     # --- Compare MLP (W_in / ff_proj) ---
#     # HF Linear layers are [out, in]. TL stores them as [in, out].
#     # So we must transpose HF weights to compare.
#     hf_in = hf_block.ff_proj.weight.T 
#     tl_in = tl_block.mlp.W_in.cpu()
    
#     diff_in = (hf_in - tl_in).abs().max().item()
#     if diff_in < 1e-4:
#         print(f"✅ MLP W_in matches (Max Diff: {diff_in:.6f})")
#     else:
#         print(f"❌ MLP W_in MISMATCH (Max Diff: {diff_in:.6f})")
    
#     # --- Compare Output/Unembed ---
#     print("\nChecking Unembedding (W_U)...")
#     # LLaDA uses 'ff_out' as the head
#     hf_u = base.ff_out.weight.T
#     tl_u = tl_model.unembed.W_U.cpu()
    
#     diff_u = (hf_u - tl_u).abs().max().item()
#     if diff_u < 1e-4:
#         print(f"✅ Unembed W_U matches (Max Diff: {diff_u:.6f})")
#     else:
#         print(f"❌ Unembed W_U MISMATCH (Max Diff: {diff_u:.6f})")

# # Run the audit
# compare_weights(tl_model)

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
        #print(f"Step {step}")
        # Calculate how many tokens we are allowed to keep this round
        # Linear schedule: At step 0, we mask 100%. At step 64, we mask 0%.
        progress = (step + 1) / steps
        tokens_to_keep_ratio = progress
        
        # A. Run the model (Forward Pass)
        # We use run_with_cache if you want to inspect, otherwise just model()
        #print(f"Input_ids: {input_ids}")
        logits = model(input_ids) 
        
        # Focus only on the generated part (ignore the prompt part)
        gen_logits = logits[0, num_prompt:] 

        #BAN COMMA
        # comma_id = model.to_single_token(",") # Usually 11
        # gen_logits[:, comma_id] = -float("inf")
        
        # B. Get Predictions and Confidence Scores
        probs = F.softmax(gen_logits, dim=-1)

        temperature = 0.1  # Try 0.1 (very strict) to 0.9 (mostly standard)
        scaled_probs = t.pow(probs, 1.0 / temperature)
        scaled_probs = scaled_probs / scaled_probs.sum(dim=-1, keepdim=True)

        predicted_ids = t.multinomial(probs, num_samples=1).squeeze(-1)
        confidence = t.gather(probs, -1, predicted_ids.unsqueeze(-1)).squeeze(-1)
        #print("Predicted tokens: ", model.tokenizer.decode(predicted_ids))
        #print("Confidence: ", confidence)
        
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
# def fix_unembed_layer(tl_model, hf_model):
#     print("🕵️ Hunting for the correct Unembedding Weights...")
    
#     # 1. Define what we are looking for
#     # TL Unembed shape is [d_model, vocab_size] (e.g., 4096, 126464)
#     target_shape = tl_model.unembed.W_U.shape
#     vocab_size = target_shape[1]
#     d_model = target_shape[0]
    
#     print(f"   Target Shape (W_U): {target_shape} (d_model={d_model}, vocab={vocab_size})")

#     candidates = []

#     # 2. Search recursively
#     for name, param in hf_model.named_parameters():
#         # SAFETY CHECK: Skip 1D tensors (biases, layernorms)
#         if len(param.shape) < 2:
#             continue
            
#         # Check for [vocab, d_model] (needs transpose)
#         if param.shape[0] == vocab_size and param.shape[1] == d_model:
#             candidates.append((name, param, True)) # True = Needs Transpose
            
#         # Check for [d_model, vocab] (no transpose needed)
#         elif param.shape[0] == d_model and param.shape[1] == vocab_size:
#             candidates.append((name, param, False)) # False = No Transpose

#     if not candidates:
#         print("❌ Could not find ANY layer with the right vocab size!")
#         return

#     # 3. Pick the best candidate
#     print(f"   Found {len(candidates)} candidates:")
#     best_candidate = None
    
#     for name, param, needs_T in candidates:
#         print(f"    - {name}")
#         # Prioritize 'lm_head'
#         if 'lm_head' in name:
#             best_candidate = (name, param, needs_T)
#             break # Found the gold standard, stop looking
#         # Fallback 1: 'embed_tokens' or 'wte' (Weight Tying)
#         elif ('wte' in name or 'embed_tokens' in name) and best_candidate is None:
#             best_candidate = (name, param, needs_T)
#         # Fallback 2: 'ff_out' (Rare, but possible in LLaDA custom blocks)
#         elif 'ff_out' in name and best_candidate is None: 
#              best_candidate = (name, param, needs_T)

#     if best_candidate:
#         name, param, needs_T = best_candidate
#         print(f"\n💉 Injecting weights from: {name}")
#         with torch.no_grad():
#             if needs_T:
#                 tl_model.unembed.W_U.copy_(param.T)
#             else:
#                 tl_model.unembed.W_U.copy_(param)
#         print("✅ Unembed Fixed.")
#     else:
#         print("❌ Found candidates but none matched known names. Check list above.")

# # RUN THIS
# hf_model = AutoModelForCausalLM.from_pretrained(
#         "GSAI-ML/LLaDA-8B-Base", 
#         trust_remote_code=True, 
#         torch_dtype=torch.float16,
#         device_map="cpu"
#     )
#fix_unembed_layer(tl_model, hf_model)

output = generate_llada(tl_model, "The Capital of France is", steps=5, gen_len=10)
print("\nFinal Output:", output)
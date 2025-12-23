import torch
import einops
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def unpermute_rope(tensor):
    """
    LLaMA stores RoPE weights as [x1, y1, x2, y2] (Interleaved).
    TransformerLens expects [x1, x2, y1, y2] (Split Halves).
    We must permute the weights to match TL's expectation.
    """
    # This rearranges the last dimension (d_head)
    return einops.rearrange(
        tensor,
        "... (pairs two) -> ... (two pairs)",
        two=2
    )

def convert_llada_weights(hf_model, cfg: HookedTransformerConfig):
    print(hf_model)
    state_dict = {}
    base_model = hf_model.model.transformer
    
    # 1. Embeddings
    state_dict["embed.W_E"] = base_model.wte.weight
    
    # 2. Iterate Layers
    for l in range(cfg.n_layers):
        hf_block = base_model.blocks[l]
        prefix = f"blocks.{l}"

        # --- Norms ---
        state_dict[f"{prefix}.ln1.w"] = hf_block.attn_norm.weight
        state_dict[f"{prefix}.ln2.w"] = hf_block.ff_norm.weight

        # --- Attention ---
        # Helper to reshape HF [Heads*HeadDim, D_Model] -> TL [Heads, D_Model, HeadDim]
        def map_qkv(weight):
            return einops.rearrange(
                weight, 
                "(n h) m -> n m h", 
                n=cfg.n_heads, 
                h=cfg.d_head
            )

        # A. Get Raw Weights & Reshape
        q = map_qkv(hf_block.q_proj.weight)
        k = map_qkv(hf_block.k_proj.weight)
        v = map_qkv(hf_block.v_proj.weight)

        # B. CRITICAL FIX: Un-permute RoPE for Q and K
        # TransformerLens requires split halves for Rotary, HF gives interleaved pairs.
        # We DO NOT touch V (V is not rotated).
        q = unpermute_rope(q)
        k = unpermute_rope(k)
        
        # C. Assign to State Dict
        state_dict[f"{prefix}.attn.W_Q"] = q
        state_dict[f"{prefix}.attn.W_K"] = k
        state_dict[f"{prefix}.attn.W_V"] = v
        
        # D. Output Projection
        # HF: [D_Model, Heads*HeadDim] -> TL: [Heads, HeadDim, D_Model]
        state_dict[f"{prefix}.attn.W_O"] = einops.rearrange(
            hf_block.attn_out.weight, 
            "m (n h) -> n h m", 
            n=cfg.n_heads
        )

        # --- MLP ---
        # NOTE on LLaMA/LLaDA Mapping:
        # LLaMA 'gate_proj' (or ff_proj) -> TL 'W_gate' (The one with SiLU)
        # LLaMA 'up_proj'                -> TL 'W_in'   (The value projection)
        # LLaMA 'down_proj' (or ff_out)  -> TL 'W_out'  (The output)
        
        state_dict[f"{prefix}.mlp.W_in"]   = hf_block.up_proj.weight.T   # Swapped from your version
        state_dict[f"{prefix}.mlp.W_gate"] = hf_block.ff_proj.weight.T   # Swapped from your version
        state_dict[f"{prefix}.mlp.W_out"]  = hf_block.ff_out.weight.T

    # 3. Final Norm & Unembed
    state_dict["ln_final.w"] = base_model.ln_f.weight
    
    # LLaDA output head is usually at the end of the transformer or passed separately
    # Assuming base_model.ff_out is the LM Head:
    state_dict["unembed.W_U"] = base_model.ff_out.weight.T

    return state_dict
import torch
import einops
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def unpermute_rope(tensor, n_heads, d_head):
    """
    LLaMA stores RoPE weights as [x1, x2, x3, x4] (adjacent pairs).
    TransformerLens expects [x1, x3, x2, x4] (split halves).
    We must permute the weights to match TL's expectation.
    """
    # tensor shape: [n_heads, d_head, d_model] or [n_heads, d_model, d_head]
    # We only operate on the d_head dimension.
    return einops.rearrange(
        tensor,
        "... (pairs two) -> ... (two pairs)",
        two=2
    )

def convert_llada_weights(hf_model, cfg: HookedTransformerConfig):
    state_dict = {}
    base_model = hf_model.model.transformer
    
    # Embeddings
    state_dict["embed.W_E"] = base_model.wte.weight
    
    for l in range(cfg.n_layers):
        hf_block = base_model.blocks[l]
        prefix = f"blocks.{l}"

        # Norms
        state_dict[f"{prefix}.ln1.w"] = hf_block.attn_norm.weight
        state_dict[f"{prefix}.ln2.w"] = hf_block.ff_norm.weight

        # --- Attention ---
        # 1. Reshape to [n_heads, d_model, d_head]
        def map_qkv(weight):
            return einops.rearrange(weight, "(n h) m -> n m h", n=cfg.n_heads, h=cfg.d_head)

        # 2. Get Raw Weights
        q = map_qkv(hf_block.q_proj.weight)
        k = map_qkv(hf_block.k_proj.weight)
        v = map_qkv(hf_block.v_proj.weight)

        # 3. FIX: Un-permute RoPE for Q and K (Rotary is only applied to Q and K)
        # We assume d_head is the LAST dimension here.
        q = unpermute_rope(q, cfg.n_heads, cfg.d_head)
        k = unpermute_rope(k, cfg.n_heads, cfg.d_head)
        
        # 4. Assign
        state_dict[f"{prefix}.attn.W_Q"] = q
        state_dict[f"{prefix}.attn._W_K"] = k  # Use underscore!
        state_dict[f"{prefix}.attn._W_V"] = v  # V is NOT rotated, so no unpermute needed
        
        state_dict[f"{prefix}.attn.W_O"] = einops.rearrange(hf_block.attn_out.weight, "m (n h) -> n h m", n=cfg.n_heads)

        # --- MLP ---
        state_dict[f"{prefix}.mlp.W_in"] = hf_block.ff_proj.weight.T
        state_dict[f"{prefix}.mlp.W_gate"] = hf_block.up_proj.weight.T
        state_dict[f"{prefix}.mlp.W_out"] = hf_block.ff_out.weight.T

    # Final & Unembed
    state_dict["ln_final.w"] = base_model.ln_f.weight
    state_dict["unembed.W_U"] = base_model.ff_out.weight.T

    return state_dict
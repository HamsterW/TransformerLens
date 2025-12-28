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
    
    # --- Embeddings ---
    state_dict["embed.W_E"] = base_model.wte.weight
    
    # NEW: Try to load Positional Embeddings (W_pos)
    # Standard models (GPT-2) have this. LLaMA/LLaDA DOES NOT.
    # We try to load it if it exists; otherwise, TL will use random initialization.
    if hasattr(base_model, "wpe"):
        print("Found W_pos (wpe), loading it...")
        state_dict["pos_embed.W_pos"] = base_model.wpe.weight
    else:
        print("⚠️ Warning: No W_pos found in HF model (Normal for LLaMA/LLaDA).")
        print("   TransformerLens will use random/zero positional embeddings.")
    
    # --- Iterate Layers ---
    for l in range(cfg.n_layers):
        hf_block = base_model.blocks[l]
        prefix = f"blocks.{l}"

        # --- Norms ---
        state_dict[f"{prefix}.ln1.w"] = hf_block.attn_norm.weight
        state_dict[f"{prefix}.ln2.w"] = hf_block.ff_norm.weight

        # --- Attention ---
        # Helper to reshape: [Heads*HeadDim, D_Model] -> [Heads, D_Model, HeadDim]
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

        # B. REMOVED: unpermute_rope 
        # For standard attention, we load Q and K exactly as they are.
        # q = unpermute_rope(q)  <-- DELETED
        # k = unpermute_rope(k)  <-- DELETED
        
        # C. Assign
        state_dict[f"{prefix}.attn.W_Q"] = q
        state_dict[f"{prefix}.attn.W_K"] = k
        state_dict[f"{prefix}.attn.W_V"] = v
        
        state_dict[f"{prefix}.attn.W_O"] = einops.rearrange(
            hf_block.attn_out.weight, 
            "m (n h) -> n h m", 
            n=cfg.n_heads
        )

        # --- MLP ---
        state_dict[f"{prefix}.mlp.W_in"]   = hf_block.up_proj.weight.T
        state_dict[f"{prefix}.mlp.W_gate"] = hf_block.ff_proj.weight.T
        state_dict[f"{prefix}.mlp.W_out"]  = hf_block.ff_out.weight.T

    # --- Final Norm & Unembed ---
    state_dict["ln_final.w"] = base_model.ln_f.weight
    
    # Check for lm_head vs ff_out
    if hasattr(hf_model, "lm_head"):
        state_dict["unembed.W_U"] = hf_model.lm_head.weight.T
    elif hasattr(base_model, "ff_out"):
        state_dict["unembed.W_U"] = base_model.ff_out.weight.T

    return state_dict
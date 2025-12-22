import torch
import einops
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def convert_llada_weights(hf_model, cfg: HookedTransformerConfig):
    """
    Converts LLaDA weights from HuggingFace format to TransformerLens format.
    """
    state_dict = {}

    # 1. Embeddings
    # LLaDA: model.transformer.wte
    state_dict["embed.W_E"] = hf_model.model.transformer.wte.weight
    
    # Note: LLaDA uses Rotary Embeddings (RoPE), which are functional. 
    # There is no "pos_embed.W_pos" weight to load.

    # 2. Iterate over layers
    # LLaDA stores blocks in: model.transformer.blocks
    blocks = hf_model.model.transformer.blocks

    for l in range(cfg.n_layers):
        hf_block = blocks[l]
        prefix = f"blocks.{l}"

        # --- Layer Norms (RMSNorm) ---
        # LLaDA uses 'attn_norm' (Pre-Attention) and 'ff_norm' (Pre-MLP)
        # Note: RMSNorm only has weights (gain), no bias.
        state_dict[f"{prefix}.ln1.w"] = hf_block.attn_norm.weight
        state_dict[f"{prefix}.ln2.w"] = hf_block.ff_norm.weight

        # --- Attention ---
        # LLaDA uses separate Linear layers: q_proj, k_proj, v_proj
        # HF Linear Weight shape: [out_features, in_features] -> [n_heads * d_head, d_model]
        # TL Target shape: [n_heads, d_model, d_head]
        
        # Helper to reshape Q, K, V
        def map_qkv(weight):
            return einops.rearrange(
                weight, 
                "(n h) m -> n m h", 
                n=cfg.n_heads, 
                h=cfg.d_head  # or m=cfg.d_model
            )

        state_dict[f"{prefix}.attn.W_Q"] = map_qkv(hf_block.q_proj.weight)
        state_dict[f"{prefix}.attn.W_K"] = map_qkv(hf_block.k_proj.weight)
        state_dict[f"{prefix}.attn.W_V"] = map_qkv(hf_block.v_proj.weight)

        # Output Projection
        # LLaDA name: attn_out
        # HF Linear Weight shape: [d_model, n_heads * d_head]
        # TL Target shape: [n_heads, d_head, d_model]
        state_dict[f"{prefix}.attn.W_O"] = einops.rearrange(
            hf_block.attn_out.weight,
            "m (n h) -> n h m",
            n=cfg.n_heads
        )

        # --- MLP (SwiGLU) ---
        # LLaDA has 3 layers: 
        #   up_proj (Input -> Hidden)
        #   ff_proj (Gate -> Hidden)
        #   ff_out  (Hidden -> Output)
        
        # TL expects weights in [in, out] format.
        # HF Linear weights are [out, in], so we simply Transpose (.T) them.

        # W_in corresponds to 'up_proj' (The signal)
        state_dict[f"{prefix}.mlp.W_in"] = hf_block.up_proj.weight.T
        
        # W_gate corresponds to 'ff_proj' (The gate)
        state_dict[f"{prefix}.mlp.W_gate"] = hf_block.ff_proj.weight.T
        
        # W_out corresponds to 'ff_out' (The down projection)
        state_dict[f"{prefix}.mlp.W_out"] = hf_block.ff_out.weight.T


    # 3. Final Norm
    state_dict["ln_final.w"] = hf_model.model.transformer.ln_f.weight

    # 4. Unembedding
    # LLaDA seems to use a separate linear head called 'ff_out' at the model root
    # HF Weight: [vocab, d_model] -> Transpose to [d_model, vocab] for TL
    state_dict["unembed.W_U"] = hf_model.model.transformer.ff_out.weight.T

    return state_dict
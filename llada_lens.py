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

from llada_generate import generate

MODEL = 'GSAI-ML/LLaDA-8B-Instruct'
SHOW_STEPS = False
SKIP_SPECIAL_TOKENS = False

device = t.device("cuda" if t.cuda.is_available() else "cpu")

tl_model = HookedTransformer.from_pretrained(
    MODEL, 
    trust_remote_code=True,
    n_devices=1,
    dtype="float16",
)

tokenizer = tl_model.tokenizer
if tokenizer.padding_side != 'left':
    tokenizer.padding_side = 'left'

# If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
assert tokenizer.pad_token_id != 126336

prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",]

# Add special tokens for the Instruct model. The Base model does not require the following two lines.
messages = [{"role": "user", "content": prompt} for prompt in prompts]
prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

encoded_outputs = tokenizer(
    prompts,
    add_special_tokens=False,
    padding=True,
    return_tensors="pt"
)
input_ids = encoded_outputs['input_ids'].to(device)
attention_mask = encoded_outputs['attention_mask'].to(device)

out = generate(tl_model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence', device=device, show_steps=SHOW_STEPS, skip_special_tokens=SKIP_SPECIAL_TOKENS)
output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=SKIP_SPECIAL_TOKENS)
print("")
for o in output:
    print(o)
    print('-' * 50)

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

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# print(gpt2_small.cfg)

# tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
# hf_llada = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=t.bfloat16)
# print(hf_llada)
tl_model = HookedTransformer.from_pretrained(
    'GSAI-ML/LLaDA-8B-Base', 
    #tokenizer=tokenizer,
    trust_remote_code=True
)
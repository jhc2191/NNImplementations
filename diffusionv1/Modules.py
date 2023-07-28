import math
from inspect import isfunction
from functools import partial

#matplotlib inline
import matplotlib.pyplot as plt
#from tqdm.auto import tqdm
#from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F


###Want to add an attention layer in here, Phil Wang does this, and it goes before the residual
###connection in the UNet. Will likely update this later



##Use basically same thing as Transformers, take in (batch_size, 1) --> (batch_size, embed_dim)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, time):
        device = time.device
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

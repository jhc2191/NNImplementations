from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from modules.py import ResnetBlock, AttnBlock

class Encoder(nn.Module):

    def __init__(self, intermediate_dim, channel_multiplier, input_dim, embed_dim):
        super().__init__()

        


        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(intermediate_dim, intermediate_dim)
        self.mid.attn_1 = AttnBlock(intermediate_dim)
        self.mid.block_2 = ResnetBlock(intermediate_dim, intermediate_dim)

        # Map to embedding space with a $3 \times 3$ convolution
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=intermediate_dim, eps=1e-6)
        self.conv_out = nn.Conv2d(intermediate_dim, 2 * embed_dim, 3, stride=1, padding=1)

    def forward():
        jhk


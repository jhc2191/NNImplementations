from einops import rearrange
import torch
from torch import nn



class Block(nn.Module):
    def __init__(self, embed_dim, out_dim, groups = 8):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, out_dim, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, embed_dim, out_dim, *, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_dim)))
        self.block1 = Block(embed_dim, out_dim, groups=groups)
        self.block2 = Block(out_dim, out_dim, groups=groups)
        self.res_conv = nn.Conv2d(embed_dim, out_dim, 1)

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.mlp(time_emb)
        h = rearrange(time_emb, "b c -> b c 1 1") + h
        h = self.block2(h)
        return h + self.res_conv(x)
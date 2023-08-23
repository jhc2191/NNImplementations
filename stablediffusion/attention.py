from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

class MiniTransformer(nn.Module):
    def __init__(self, intermediate_dim, num_heads, num_layers, embed_dim):
        super().__init__()
        self.norm = torch.nn.GroupNorm(32, intermediate_dim)
        self.proj_in = nn.Conv2d(intermediate_dim, intermediate_dim)

        self.transformer_block = nn.ModuleList([TransformerBlock(intermediate_dim, num_heads, intermediate_dim // num_heads, embed_dim) for _ in range(num_layers)])

        self.proj_out = nn.Conv2d(intermediate_dim, intermediate_dim)
    
    def forward(self, x, embeds):
        batch_size, intermediate_dim, h, w = x.shape
        x_residual = x
        x = self.norm(x)
        x = self.proj_in(x)


        # Transpose and reshape from `[batch_size, channels, height, width]`
        # to `[batch_size, height * width, channels]`
        x = x.permute(0, 2, 3, 1).view(batch_size, h * w, intermediate_dim)

        for block in self.transformer_blocks:
            x = block(x, embeds)

        # Reshape and transpose from `[batch_size, height * width, channels]`
        # to `[batch_size, channels, height, width]`
        x = x.view(batch_size, h, w, intermediate_dim).permute(0, 3, 1, 2)

        x = self.proj_out(x)
        return x + x_residual 


class TransformerBlock(nn.Module):

    def __init__(self, intermediate_dim, num_heads, head_dim, embed_dim):
        super().__init__()

        self.attn1 = CrossAttention(intermediate_dim, intermediate_dim, num_heads, head_dim)
        self.norm1 = nn.LayerNorm(intermediate_dim)

        self.attn2 = CrossAttention(intermediate_dim, embed_dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(intermediate_dim)

        self.ff = FeedForward(intermediate_dim)
        self.norm3 = nn.LayerNorm(intermediate_dim)
    
    def forward(self, x, embeds):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), embeds=embeds) + x
        x = self.ff(self.norm3(x)) + x
        return x


class CrossAttention(nn.Module):
    def __init__(self, intermediate_dim, num_heads, head_dim, embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim **-0.5

        attn_dim = num_heads * head_dim
        self.q = nn.Linear(intermediate_dim, attn_dim)
        self.k = nn.Linear(intermediate_dim, attn_dim)
        self.v = nn.Linear(intermediate_dim, attn_dim)

        self.out = nn.Sequential(nn.Linear(attn_dim, intermediate_dim))
    
    def forward(self, x, embeds):
        if embeds is None:
            embeds = x
        
        batch_size, len_q, len_k, len_v= q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.q(x).view(batch_size, len_q, self.num_heads, self.head_dim)
        k = self.k(embeds).view(batch_size, len_k, self.num_heads, self.head_dim)
        v = self.v(embeds).view(batch_size, len_v, self.num_heads, self.head_dim)

        attention_values = torch.matmul(q / self.scale, k.transpose(2, 3))
        attention_values = self.dropout(F.softmax(attention_values, dim=-1))
        q = torch.matmul(attention_values, v)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        return self.out(q)




class FeedForward(nn.Module):
    def __init__(self, intermediate_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(intermediate_dim, intermediate_dim * 4),
            nn.Dropout(0.),
            nn.Linear(intermediate_dim * 4, intermediate_dim)
        )

    def forward(self, x):
        return self.net(x)
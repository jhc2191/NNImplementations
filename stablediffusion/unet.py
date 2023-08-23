import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stablediffusion.attention import MiniTransformer

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, intermediate_dim, num_res_blocks, num_heads, attn_levels, transformer_layers, dim_multipliers, embed_dim):
        super().__init__()
        self.intermediate_dim = intermediate_dim

        levels = len(dim_multipliers)
        self.time_embed = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim * 4),
            nn.SiLU(),
            nn.Linear(intermediate_dim * 4, intermediate_dim * 4),
        )

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(TimestepEmbedSequential(
            nn.Conv2d(in_dim, intermediate_dim, 3)))
        input_block_dim = [intermediate_dim]
        dim_list = [intermediate_dim * m for m in dim_multipliers]
        for i in range(levels):
            for _ in range(num_res_blocks):
                layers = [ResBlock(intermediate_dim, embed_dim, out_dim=dim_list[i])]
                intermediate_dim = dim_list[i]

                if i in attn_levels:
                    layers.append(MiniTransformer(intermediate_dim, num_heads, transformer_layers, embed_dim))

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_dim.append(intermediate_dim)
            
            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(intermediate_dim)))
                input_block_dim.append(intermediate_dim)


        self.middle_block = TimestepEmbedSequential(
            ResBlock(intermediate_dim, intermediate_dim*4),
            MiniTransformer(intermediate_dim, num_heads, transformer_layers, embed_dim),
            ResBlock(intermediate_dim, intermediate_dim*4),
        )
        self.output_blocks = nn.ModuleList([])
        for i in reversed(range(levels)):
            for j in range(num_res_blocks + 1):
                layers = [ResBlock(intermediate_dim + input_block_dim.pop(), embed_dim, out_dim=dim[i])]
                intermediate_dim = dim_list[i]
                # Add transformer
                if i in attn_levels:
                    layers.append(MiniTransformer(intermediate_dim, num_heads, transformer_layers, embed_dim))

                if i != 0 and j == num_res_blocks:
                    layers.append(UpSample(intermediate_dim))

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, intermediate_dim),
            nn.SiLU(),
            nn.Conv2d(intermediate_dim, out_dim, 3),
        )

    def time_step_embedding(self, time_steps, max_period=10000):
        ##from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/stable_diffusion/model/unet.py
        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self.channels // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def forward(self, x: torch.Tensor, time_steps, embeds):
        x_input_block = []

        time_embeds = self.time_step_embedding(time_steps)
        time_embeds = self.time_embed(time_embeds)

        for module in self.input_blocks:
            x = module(x, time_embeds, embeds)
            x_input_block.append(x)

        x = self.middle_block(x, time_embeds, embeds)

        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, time_embeds, embeds)

        return self.out(x)


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, embeds):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, embeds)
            elif isinstance(layer, MiniTransformer):
                x = layer(x)
            else:
                x = layer(x)
        return x

class UpSample(nn.Module):
    def __init__(self, intermediate_dim):
        super().__init__()
        self.conv = nn.Conv2d(intermediate_dim, intermediate_dim, 3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, intermediate_dim):
        super().__init__()
        self.conv = nn.Conv2d(intermediate_dim, intermediate_dim, 3, stride=2)

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, intermediate_dim, t_emb_dim, out_dim):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, intermediate_dim),
            nn.SiLU(),
            nn.Conv2d(intermediate_dim, out_dim, 3),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_dim),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_dim, out_dim, 3)
        )
        self.skip_connection = nn.Conv2d(intermediate_dim, out_dim)

    def forward(self, x, t_embeds):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        h = self.in_layers(x)
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        h = h + t_emb[:, :, None, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h
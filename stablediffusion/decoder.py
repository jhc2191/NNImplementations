from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from stablediffusion.modules import ResnetBlock, AttnBlock


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3)

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2)
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, intermediate_dim, block_multipliers, num_resnet_layers, out_dim, embed_dim):
        super().__init__()
        num_blocks= len(block_multipliers)

        block_list = [m * intermediate_dim for m in block_multipliers]
        intermediate_dim = block_list[-1]
        self.conv_in = nn.Conv2d(out_dim, intermediate_dim, 3)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(intermediate_dim, intermediate_dim)
        self.mid.attn_1 = AttnBlock(intermediate_dim)
        self.mid.block_2 = ResnetBlock(intermediate_dim, intermediate_dim)

        self.up = nn.ModuleList()
        for i in reversed(range(num_blocks)):
            resnet_blocks = nn.ModuleList()
            for _ in range(num_resnet_layers + 1):
                resnet_blocks.append(ResnetBlock(intermediate_dim, block_list[i]))
                channels = block_list[i]

            up = nn.Module()
            up.block = resnet_blocks
            up.upsample = UpSample(intermediate_dim)
            # Prepend to be consistent with the checkpoint
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, intermediate_dim)
        self.conv_out = nn.Conv2d(intermediate_dim, out_dim, 3)

    def forward(self, x):
        x = self.conv_in(x)

        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        for up in reversed(self.up):
            for block in up.block:
                x = block(x)
            x = up.upsample(x)

        x = self.norm_out(x)
        x = x*torch.sigmoid(x)
        img = self.conv_out(x)

        return img
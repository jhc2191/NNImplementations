from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from stablediffusion.modules import ResnetBlock, AttnBlock

class Encoder(nn.Module):

    def __init__(self, intermediate_dim, block_multipliers, input_dim, num_resnet_layers, embed_dim):
        super().__init__()

        num_blocks = len(block_multipliers)
        self.conv_in = nn.Conv2d(input_dim, intermediate_dim, 3)

        block_list = [m * intermediate_dim for m in [1] + block_multipliers]

        self.down = nn.ModuleList()
        for i in range(num_blocks):
            resnet_blocks = nn.ModuleList()
            for _ in range(num_resnet_layers):
                resnet_blocks.append(ResnetBlock(intermediate_dim, block_list[i + 1]))
                intermediate_dim = block_list[i + 1]
            
            down = nn.Module()
            down.block = resnet_blocks
            down.downsample = DownSample(intermediate_dim)
            self.down.append(down)


        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(intermediate_dim, intermediate_dim)
        self.mid.attn_1 = AttnBlock(intermediate_dim)
        self.mid.block_2 = ResnetBlock(intermediate_dim, intermediate_dim)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=intermediate_dim, eps=1e-6)
        self.conv_out = nn.Conv2d(intermediate_dim, 2 * embed_dim, 3, stride=1, padding=1)

    def forward(self, img):
        x = self.conv_in(img)

        for down in self.down:
            for block in down.block:
                x = block(x)
            x = down.downsample(x)

        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        x = self.norm_out(x)
        x = x * torch.sigmoid(x)
        x = self.conv_out(x)

        return x


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)
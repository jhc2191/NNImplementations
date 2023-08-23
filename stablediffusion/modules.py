from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class AttnBlock(nn.Module):

    def __init__(self, intermediate_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=intermediate_dim, eps=1e-6)

        self.q = nn.Linear(intermediate_dim, intermediate_dim)
        self.k = nn.Linear(intermediate_dim, intermediate_dim)
        self.v = nn.Linear(intermediate_dim, intermediate_dim)

        self.scale = intermediate_dim** -0.5
        self.fc_o = nn.Linear(intermediate_dim, intermediate_dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        batch_size, temp_dim, height, width= x.size

        q = self.fc_q(q).view(batch_size, temp_dim, height*width)
        k = self.fc_k(k).view(batch_size, temp_dim, height*width)
        v = self.fc_v(v).view(batch_size, temp_dim, height*width)

        ##do the dot proudct and scale
        attention_values = torch.matmul(q / self.scale, k.transpose(2, 3))
        attention_values = F.softmax(attention_values, dim=2)
        out = torch.matmul(attention_values, v)

        out += residual

        return out


class ResnetBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=input_dim, eps=1e-6)
        self.conv1 = nn.Conv2d(input_dim, output_dim, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=output_dim, eps=1e-6)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 3, stride=1, padding=1)
        self.residual = nn.Conv2d(input_dim, output_dim, 1, stride=1, padding=0)
    
    def forward(self, x):
        residual = x

        x = self.norm1(x)
        x = x*torch.sigmoid(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = x*torch.sigmoid(x)
        x = self.conv2(x)

        return self.residual(residual) + x
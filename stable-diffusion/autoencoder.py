from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, emb_dim, z_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = nn.Conv2d(2 * z_dim, 2 * emb_dim, 1)
        self.post_quant_conv = nn.Conv2d(emb_dim, z_dim, 1)

    def encode(self, x) -> 'GaussianDistribution':
        z = self.encoder(x)
        moments = self.quant_conv(z)
        return GaussianDistribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)
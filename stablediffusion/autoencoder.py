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

class GaussianDistribution:
    """
    ## Gaussian Distribution from 
    # https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/stable_diffusion/model/autoencoder.py#L253
    """

    def __init__(self, parameters: torch.Tensor):
        """
        :param parameters: are the means and log of variances of the embedding of shape
            `[batch_size, z_channels * 2, z_height, z_height]`
        """
        # Split mean and log of variance
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        # Clamp the log of variances
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self):
        # Sample from the distribution
        return self.mean + self.std * torch.randn_like(self.std)
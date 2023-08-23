import torch
from torch import nn
from diffusionv1.Blocks import Block, ResnetBlock
from diffusionv1.Modules import SinusoidalPositionEmbeddings

class Unet(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.init_dim = init_dim

        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        ###standard diffusion architecture to downsample, have an intermediate bottleneck
        ###then re-upsample, nothing too out of chracter

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                        ##would be attention layer
                        nn.Conv2d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ] 
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        ##would be attention layer
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        ##would be attention layer
                        nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity(),
                    ] 
                )
            )

        self.out_dim=out_dim
        self.final_conv = nn.Sequential(
            ResnetBlock(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) 

        h = []

        # downsample
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            ##would be attention layer
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        ##would be attention layer
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            ##would be attention layer
            x = upsample(x)
        x=self.final_conv(x)
        return x
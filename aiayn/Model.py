import torch
import torch.nn as nn
import numpy as np
from aiayn.Encoder import Encoder
from aiayn.Decoder import Decoder
 
###Positional Encoding taken from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, src_pad_idx, trg_pad_idx, encoding_dim=512, model_dim=512, intermediate_dim=2048,
            num_layers=6, num_heads=8, k_dim=64, v_dim=64, dropout=0.1, n_position=200):
         super().__init__()
         self.encoder = Encoder(input_dim, encoding_dim, model_dim, intermediate_dim, num_layers, num_heads, k_dim, v_dim, dropout, n_position)
         self.decoder = Decoder(output_dim, encoding_dim, model_dim, intermediate_dim, num_layers, num_heads, k_dim, v_dim, dropout, n_position)
         self.src_pad_idx = src_pad_idx
         self.trg_pad_idx = trg_pad_idx
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output
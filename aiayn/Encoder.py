import torch
import torch.nn as nn

from aiayn.Model import PositionalEncoding
from aiayn.Modules import MultiHeadAttention, PositionWiseFeedForward

class Encoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, model_dim, intermediate_dim, num_layers, num_heads, dropout, k_dim, v_dim, n_position=200):
        super().__init__()

        self.token_embedding = nn.Embedding(input_dim, encoding_dim)
        self.position_encoding = PositionalEncoding(encoding_dim, n_position)

        self.layers = nn.ModuleList([Encoder_Layer(model_dim, intermediate_dim, num_heads, k_dim, v_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([model_dim]))

    def forward(self, src, src_mask):
        ##updated embeddings not included in original paper
        src = self.dropout(self.token_embedding(src)*self.scale(src) + self.position_encoding(src))

        for layer in self.layers:
            src = layer(src, src_mask)
    
        return src


class Encoder_Layer(nn.Module):
    def __init__(self, model_dim, intermediate_dim, num_heads, k_dim, v_dim, dropout):
        super().__init__()

        self.self_attention=MultiHeadAttention(model_dim, num_heads, k_dim, v_dim, dropout)
        self.first_layer_norm = nn.LayerNorm(model_dim)

        self.positionwise_feedforward = PositionWiseFeedForward(model_dim, intermediate_dim, dropout)
        self.second_layer_norm = nn.LayerNorm(intermediate_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        ##apply attention and feed forward layers
        src_non_residual, _ = self.self_attention(src, src, src, src_mask)
        src = self.first_layer_norm(src + self.dropout(src_non_residual))

        src_non_residual = self.positionwise_feedforward(src)
        src = self.second_layer_norm(src + self.dropout(src_non_residual))

        return src


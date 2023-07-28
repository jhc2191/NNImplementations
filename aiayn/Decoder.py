import torch
import torch.nn as nn

from aiayn.Model import PositionalEncoding
from aiayn.Modules import MultiHeadAttention, PositionWiseFeedForward


###ADD SCALING

class Decoder(nn.Module):
    def __init__(self, output_dim, encoding_dim, model_dim intermediate_dim, num_layers, num_heads, dropout, k_dim, v_dim n_position=200):
        super().__init__()
        self.token_embedding = nn.Embedding(output_dim, encoding_dim)
        self.position_encoding = PositionalEncoding(encoding_dim, n_position)

        self.layers = nn.ModuleList([Decoder_Layer(model_dim, intermediate_dim, num_heads, k_dim, v_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(intermediate_dim, output_dim)

    def forward(self, trg, encoder_output, src_mask, trg_mask):
        trg = self.token_embedding(trg)
        trg = self.dropout(self.position_encoding(trg))

        for layer in self.layers:
            trg = layer(trg, encoder_output, trg_mask, src_mask)

        return trg

class Decoder_Layer(nn.Module):
    def __init__(self, model_dim, intermediate_dim, num_heads, k_dim, v_dim, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, k_dim, v_dim, dropout=dropout)
        self.self_attention_layer_norm = nn.LayerNorm(model_dim)
        self.encoder_attention = MultiHeadAttention(model_dim, num_heads, k_dim, v_dim, dropout=dropout)
        self.encoder_attention_layer_norm = nn.LayerNorm(model_dim)
        self.positionwise_feedforward = PositionWiseFeedForward(model_dim, intermediate_dim, dropout=dropout)
        self.ff_layer_norm = nn.LayerNorm(intermediate_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, decoder_input, encoder_outpout, src_mask, trg_mask):
        decoder_output, _ = self.self_attention(decoder_input, decoder_input, decoder_input, trg_mask)
        decoder_output = self.self_attention_layer_norm(decoder_input + self.dropout(decoder_output))

        decoder_output_non_residual, _ = self.encoder_attention(encoder_outpout, decoder_input, encoder_output, trg_mask)
        decoder_output = self.encoder_attention_layer_norm(decoder_output + self.dropout(decoder_output_non_residual))

        decoder_output_non_residual = self.positionwise_feedforward(decoder_output)
        decoder_output = self.ff_layer_norm(decoder_output + self.dropout(decoder_output_non_residual))

        return decoder_output


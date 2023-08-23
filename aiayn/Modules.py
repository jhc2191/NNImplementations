import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, k_dim, v_dim, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        ###these will just map back to the same dimension
        self.fc_q = nn.Linear(model_dim, num_heads*k_dim)
        self.fc_k = nn.Linear(model_dim, num_heads*k_dim)
        self.fc_v = nn.Linear(model_dim, num_heads*v_dim)

        self.scale = torch.sqrt(torch.FloatTensor([model_dim//num_heads]))
        
        self.fc_o = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, src_mask):
        ##q, k, v have dimension [batch_size, q/k/v_len, model_dim]

        k_dim, v_dim, num_heads = self.k_dim, self.v_dim, self.num_heads
        
        batch_size, len_q, len_k, len_v= q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        ##separate the heads
        q = self.fc_q(q).view(batch_size, len_q, num_heads, k_dim)
        k = self.fc_k(k).view(batch_size, len_k, num_heads, k_dim)
        v = self.fc_v(v).view(batch_size, len_v, num_heads, v_dim)

        ##do the dot proudct and scale
        mask=mask.unsqueeze(1)

        attention_values = torch.matmul(q , k.transpose(2, 3))
        attention_values = attention_values/self.scale
        attention_values = attention_values.masked_fill(src_mask == 0, -1e9)
        attention_values = self.dropout(F.softmax(attention_values, dim=-1))
        q = torch.matmul(attention_values, v)

        ##reframe to get dimensions back to normal
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc_o(q))
        q += residual

        return q, attention_values


class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_dim, intermediate_dim, dropout):
        super().__init__()
        self.first_linear = nn.Linear(model_dim, intermediate_dim) 
        self.second_linear = nn.Linear(intermediate_dim, model_dim) 
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        residual = src
        src = self.second_linear(F.relu(self.first_linear(src)))
        src = self.dropout(src)
        src += residual
        src = self.layer_norm(src)

        return src
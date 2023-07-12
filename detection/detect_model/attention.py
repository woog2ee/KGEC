import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def forward(self, Q, K, V, mask=None, dropout=None):
        print(f'\n {Q.shape} {K.shape} {V.shape} \n')
        print(f'{mask.shape}')
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        if mask != None: scores = scores.masked_fill(mask==0, -1e9)
            
        scores = F.softmax(scores, dim=-1)
        if dropout != None: scores = dropout(scores)
            
        attn = torch.matmul(scores, V)
        return attn, scores


class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads, d_model, dropout):
    super().__init__()

    assert d_model % n_heads == 0
    self.n_heads = n_heads
    self.d_model = d_model
    self.h_model = d_model // n_heads  # 32
    
    self.fc_query = nn.Linear(d_model, d_model)
    self.fc_key   = nn.Linear(d_model, d_model)
    self.fc_value = nn.Linear(d_model, d_model)
    self.fc_o     = nn.Linear(d_model, d_model)
    self.dropout  = nn.Dropout(dropout)


  def forward(self, Q, K, V, mask=None):
    batch_size = Q.size(0)
    # query: [batch_size, query_len, hidden_dim]
    # key  : [batch_size, key_len, hidden_dim]
    # value: [batch_size, value_len, hidden_dim]

    Q = self.fc_query(Q)
    K = self.fc_key(K)
    V = self.fc_value(V)
    # Q: [batch_size, query_len, hidden_dim]
    # K: [batch_size, key_len, hidden_dim]
    # V: [batch_size, value_len, hidden_dim]

    # hidden_dim to n_heads * head_dim
    Q = Q.view(batch_size, -1, self.n_heads, self.h_model).permute(0, 2, 1, 3)
    K = K.view(batch_size, -1, self.n_heads, self.h_model).permute(0, 2, 1, 3)
    V = V.view(batch_size, -1, self.n_heads, self.h_model).permute(0, 2, 1, 3)
    # Q: [batch_size, n_heads, query_len, head_dim]
    # K: [batch_size, n_heads, key_len, head_dim]
    # V: [batch_size, n_heads, value_len, head_dim]

    # attention energies, (Q*K^T/n)
    energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(Q.size(-1))
    # energy: [batch_size, n_heads, query_len, key_len]

    # masking
    if mask is not None:
      energy = energy.masked_fill(mask==0, -1e10)

    # attention score, softmax(Q*K^T/n)
    attention = torch.softmax(energy, dim=-1)
    # attention: [batch_size, n_heads, query_len, key_len]

    # softmax(Q*K^T/n)*V
    x = torch.matmul(self.dropout(attention), V)
    # x: [batch_size, n_heads, query_len, h_model]

    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(batch_size, -1, self.d_model)
    # x: [batch_size, query_len, hidden_dim]
    x = self.fc_o(x)
    return x, attention
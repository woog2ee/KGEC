import math
import torch
import torch.nn as nn



class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.44715 * torch.pow(x, 3))))
    
    
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.layer1  = nn.Linear(d_model, d_ff)
        self.layer2  = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.active  = GELU()
        
        
    def forward(self, x):
        x = self.dropout(self.active(self.layer1(x)))
        x = self.layer2(x)
        return x
    
    
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta  = nn.Parameter(torch.zeros(features))
        self.eps   = eps
        
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return (x - mean) * self.gamma / (std + self.eps) + self.beta
    
    

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        self.norm    = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))
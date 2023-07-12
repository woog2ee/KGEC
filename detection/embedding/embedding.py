import math
import torch
import torch.nn as nn
from .koreanphoneme import construct_vocab



class PositionalEmbedding(nn.Module):
    def __init__(self, emb_size, max_word_len):
        super().__init__()
        
        self.max_word_len = max_word_len

        # pe: [max_word_len, emb_size]
        pe = torch.zeros(max_word_len, emb_size).float()
        pe.require_grad = False
        
        # position: convert '1, 2, 3, ..., max_word_len' form to [max_word_len, 1]
        position = torch.arange(0, max_word_len).float().unsqueeze(1)
        
        # div_term: convert '0, 2, 4, ..., emb_size-2' form to exponential [max_word_len]
        div_term = (torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # pe: [1, max_word_len, emb_size]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        
    def forward(self):
        # [1, max_word_len, emb_size] / [1, 150, 100]
        return self.pe[:, :self.max_word_len]
    
    
    
class CharAwareEmbedding(nn.Module):
    def __init__(self, embed_size, hidden_dim, num_layers, bias, dropout=0.1,
                 bidirectional=True, sow_token=True, eow_token=False):
        super().__init__()
        
        self.vocab         = construct_vocab()
        self.hidden_dim    = hidden_dim
        self.bidirectional = bidirectional
        self.sow_token     = sow_token
        self.eow_token     = eow_token
        
        self.char_emb = nn.Embedding(len(self.vocab), embed_size,
                                     padding_idx=self.vocab['<PAD>'])
        
        self.lstm = nn.LSTM(input_size    = embed_size,
                            hidden_size   = hidden_dim,
                            num_layers    = num_layers,
                            bias          = bias,
                            batch_first   = True,
                            dropout       = dropout,
                            bidirectional = bidirectional)
        self.weights = nn.Parameter(torch.empty(hidden_dim*2, embed_size))
        self.register_parameter('char_weights', self.weights)
        
        self.init_weight(self.char_emb)
        self.init_weight(self.lstm)
        self.init_weight(self.weights)


    def forward(self, input):
        # input: 1 * sentence
        # input: [max_word_len, max_char_len] / [150, 20]

        char_embs = self.char_emb(input)
        # char_embs: [max_word_len, max_char_len, embed_size] / [150, 20, 100]
        
        output, h_n = self.lstm(char_embs)
        if self.bidirectional:
            forward_output, backward_output = torch.split(output, self.hidden_dim, dim=2)
            forward_last_hidden  = forward_output[:, -1, :]
            backward_last_hidden = backward_output[:, 0, :]
        # each last_hidden : [max_word_len, lstm_d_model] / [150, 256]  
            
        concat = torch.cat((forward_last_hidden, backward_last_hidden), dim=1)
        combined = concat @ self.weights
        # combined: [max_word_len, embed_size] / [150, 100]
        
        return combined
    
    
    def init_weight(self, layer):
        if 'Parameter' in str(type(layer)):
            return nn.init.uniform_(layer, -1.0, 1.0)

        elif 'Embedding' in str(type(layer)):
            return nn.init.uniform_(layer.weight, -1.0, 1.0)

        elif 'LSTM' in str(type(layer)):
            for m in layer.modules():
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
        

class KoreanWordEmbedding(nn.Module):
    def __init__(self, embed_size, hidden_dim, num_layers, bias, dropout=0.1,
                 bidirectional=True, sow_token=True, eow_token=False, max_word_len=150):
        super().__init__()
        
        self.word = CharAwareEmbedding(embed_size    = embed_size,
                                       hidden_dim    = hidden_dim,
                                       num_layers    = num_layers,
                                       bias          = bias,
                                       dropout       = dropout,
                                       bidirectional = bidirectional,
                                       sow_token     = sow_token,
                                       eow_token     = eow_token)
        self.pos = PositionalEmbedding(embed_size, max_word_len)
        self.dropout = nn.Dropout(p=dropout)

   
    def forward(self, input):
        # input: batch_size * sentence
        # input: [batch_size, max_word_len, max_char_len] / [64, 150, 20]
 
        embed = torch.stack([self.word(word) for word in input])
        # embed: [batch_size, max_word_len, embed_size] / [64, 150, 100] 

        pos = self.pos()
        pos - pos.expand(input.size(1), -1, -1)
        # pos: [batch_size, max_word_len, embed_size] / [64, 150, 100]

        embed += pos
        return self.dropout(embed)
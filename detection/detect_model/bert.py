import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .utils import FeedForward
from embedding import KoreanWordEmbedding



class EncoderBlock(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout):
        super().__init__()
        
        self.multihead_attn = MultiHeadAttention(n_heads, d_model, dropout)
        self.feedforward    = FeedForward(d_model, d_ff, dropout)
       
        self.layernorm_attn = nn.LayerNorm(d_model)
        self.layernorm_ff   = nn.LayerNorm(d_model)
        self.dropout        = nn.Dropout(dropout)
        
        
    def forward(self, x, mask):
        x_, _ = self.multihead_attn(x, x, x, mask)

        x = self.layernorm_attn(x + self.dropout(x_))

        x_ = self.feedforward(x)

        x = self.layernorm_ff(x + self.dropout(x_))
        return x
    
    
    
class BERT(nn.Module):
    def __init__(self, n_heads, bert_d_model, bert_layers, bert_dropout,
                 embed_size, lstm_d_model, lstm_layers, lstm_bias, lstm_dropout,
                 sow_token, eow_token):
        super().__init__()
        
        self.n_heads  = n_heads
        self.d_model  = bert_d_model
        self.n_layers = bert_layers
        self.d_ff     = bert_d_model * 4
        
        self.embedding = KoreanWordEmbedding(embed_size    = embed_size,
                                             hidden_dim    = lstm_d_model,
                                             num_layers    = lstm_layers,
                                             bias          = lstm_bias,
                                             dropout       = lstm_dropout,
                                             bidirectional = True,
                                             sow_token     = sow_token,
                                             eow_token     = eow_token)
        
        self.encoder_blocks = nn.ModuleList([EncoderBlock(n_heads, bert_d_model, self.d_ff, bert_dropout)
                                             for _ in range(bert_layers)])
        

    def make_mask(self, x):
        x_ = torch.sum(x, dim=2)
        mask = (x_ != 0).unsqueeze(1).unsqueeze(2)
        return mask

        
    def forward(self, x):
        # x: [batch_size, max_word_len, max_char_len] / [64, 150, 20]
        #print(f'\n input: {x.shape}\n')

        mask = self.make_mask(x)
        # mask: [batch_size, 1, 1, max_word_len] / [64, 1, 1, 150]
        #print(f'\n mask: {mask.shape}\n')

        x = self.embedding(x)
        # x: [batch_size, max_word_len, embed_size] / [64, 150, 256]
        #print(f'\n emb: {x.shape}\n')

        for encoder_block in self.encoder_blocks:
            x = encoder_block.forward(x, mask)
            
        return x
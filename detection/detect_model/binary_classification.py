import torch.nn as nn
from .bert import BERT



class BERTBinaryClassification(nn.Module):
    def __init__(self, bert=BERT):
        super().__init__()
        
        self.bert = bert
        self.binary_classify = BinaryClassification(self.bert.d_model)
        
    
    def forward(self, x):
        x = self.bert(x)
        return self.binary_classify(x)
    
    
    
class BinaryClassification(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.layer   = nn.Linear(hidden_dim, 2)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        
    def forward(self, x):
        return self.softmax(self.layer(x))
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel



tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# print(BartModel.from_pretrained('facebook/bart-large'))

class BARTClassifier(nn.Module):
    def __init__(self):
        super(BARTClassifier ,self).__init__()
        self.bart = BartModel.from_pretrained('facebook/bart-large')
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(in_features = 1024, out_features = 4)

        nn.init.normal_(self.cls.weight, std = 0.2)
        nn.init.normal_(self.cls.bias, 0)

    def forward(self, x):
        bart_out = self.bart(x)
        vec_0 = output[0]  
        vec_0 = vec_0[:, 0, :] 
        vec_0 = vec_0.view(-1, 1024)
        
        vec_0 = self.dropout(vec_0)
        result = self.cls(vec_0)
        
        return result



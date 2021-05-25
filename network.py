import numpy as np
import torch.nn as nn
import torch
from transformers import AutoModel
import torch.nn.functional as F

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)




class TransformerLayer(nn.Module):
    def __init__(self, dropout_prob=0.1,both=False,
                 pretrained_path='aubmindlab/bert-base-arabert'):
        super(TransformerLayer, self).__init__()
        self.both = both
        self.transformer = AutoModel.from_pretrained(pretrained_path)
        #for param in self.transformer.parameters():
        #    param.requires_grad = False
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.attention = AttentionWithContext(self.output_num())


    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs[1]
        pooled = self.dropout1(pooled)
        if self.both:
            output = outputs[0]
            output = self.dropout2(output)
            gattention = self.attention(output)
            return output, pooled, gattention
        else:
            return pooled

    def get_parameters(self):
        return [{"params": self.parameters(),"lr_mult":1, 'decay_mult':1}]

    def output_num(self):
        return self.transformer.config.hidden_size

class AttentionWithContext(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionWithContext, self).__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)
        self.apply(init_weights)
    def forward(self, inp):
        u = torch.tanh_(self.attn(inp))
        a = F.softmax(self.contx(u), dim=1)
        s = (a * inp).sum(1)
        return s


class Dropout_on_dim(nn.modules.dropout._DropoutNd):
    """ Dropout that creates a mask based on 1 single input, and broadcasts
    this mask accross the batch
    """

    def __init__(self, p, dim=1, **kwargs):
        super().__init__(p, **kwargs)
        self.dropout_dim = dim
        self.multiplier = 1.0 / (1.0 - self.p)

    def forward(self, X):
        mask = torch.bernoulli(X.new(X.size(self.dropout_dim)).fill_(1 - self.p))
        return X * mask * self.multiplier

class Classifier(nn.Module):
    def __init__(self, in_feature, class_num, dropout_prob=0.4):
        super(Classifier, self).__init__()
        self.num_class = class_num
        self.attention = AttentionWithContext(in_feature)
        self.ad_layer1 = nn.Linear(3 * in_feature, 512)
        self.ad_layer2 = nn.Linear(512, class_num)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.apply(init_weights)

    def forward(self, x, pooled, gattention):
        x = self.attention(x)
        #x = x + pooled
        # x = self.relu1(x)
        x = torch.cat([pooled, gattention, x], 1)
        x = self.ad_layer1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        y = self.ad_layer2(x)
        return y

    def output_num(self):
        return self.num_class

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 1}]

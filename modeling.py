import numpy as np
import torch.nn as nn
import torch
from transformers import AutoModel
import torch.nn.functional as F
import utils
from layers import AttentionWithContext, MultiHeadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TransformerLayer(nn.Module):
    def __init__(self, dropout_prob=0.15,both=False,
                 pretrained_path='aubmindlab/bert-base-arabert'):
        super(TransformerLayer, self).__init__()

        self.both = both
        self.transformer = AutoModel.from_pretrained(pretrained_path)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)


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
            return output, pooled
        else:
            return pooled

    def output_num(self):
        return self.transformer.config.hidden_size


class MTClassifier(nn.Module):
    def __init__(self, in_feature, class_num_sar=1, class_num_sent=3 , dropout_prob=0.2):
        super(MTClassifier, self).__init__()
        self.W_inter = nn.Parameter(nn.init.xavier_normal_(torch.tensor((in_feature, in_feature))))
        self.b_inter = nn.Parameter(torch.zeros(in_feature))
        self.sar_attention = AttentionWithContext(in_feature)
        self.sent_attention = AttentionWithContext(in_feature)

        self.sracasmClassifier = nn.Sequential(
            nn.Linear(3 * in_feature, in_feature),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(in_feature, class_num_sar)
        )
        self.SentimentClassifier = nn.Sequential(
            nn.Linear(3 * in_feature, in_feature),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(in_feature, class_num_sent)
        )

        self.apply(utils.init_weights)

    def forward(self, x, pooled):
        att_sar = self.sar_attention(x)
        att_sent = self.sent_attention(x)

        sar_x = att_sar.mul(torch.sigmoid(torch.matmul(att_sar, self.W_inter) + self.b_inter))
        sent_x = att_sent.mul(torch.sigmoid(torch.matmul(att_sent, self.W_inter) + self.b_inter))

        sar_xx = torch.cat([sar_x, att_sar, pooled], 1)
        sent_xx = torch.cat([sent_x, att_sent, pooled], 1)

        sar_out = self.sracasmClassifier(sar_xx)
        sent_out = self.SentimentClassifier(sent_xx)
        return sar_out, sent_out

class MTClassifier1(nn.Module):
    def __init__(self, in_feature, class_num_sar=1, class_num_sent=3 , dropout_prob=0.2):
        super(MTClassifier1, self).__init__()
        self.sar_attention = AttentionWithContext(in_feature)
        self.sent_attention = AttentionWithContext(in_feature)

        self.sracasmClassifier = nn.Sequential(
            nn.Linear(2 * in_feature, in_feature),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(in_feature, class_num_sar)
        )
        self.SentimentClassifier = nn.Sequential(
            nn.Linear(2 * in_feature, in_feature),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(in_feature, class_num_sent)
        )

        self.apply(utils.init_weights)

    def forward(self, x, pooled):
        att_sar = self.sar_attention(x)
        att_sent = self.sent_attention(x)

        sar_xx = torch.cat([att_sar, pooled], 1)
        sent_xx = torch.cat([att_sent, pooled], 1)

        sar_out = self.sracasmClassifier(sar_xx)
        sent_out = self.SentimentClassifier(sent_xx)
        return sar_out, sent_out

class MTClassifier2(nn.Module):
    def __init__(self, in_feature, class_num_sar=1, class_num_sent=3 , dropout_prob=0.2):
        super(MTClassifier2, self).__init__()
        self.W_inter = nn.Parameter(nn.init.xavier_normal_(torch.ones((in_feature, in_feature))))
        self.b_inter = nn.Parameter(torch.zeros(in_feature))
        self.sar_attention = AttentionWithContext(in_feature)
        self.sent_attention = AttentionWithContext(in_feature)

        self.sracasmClassifier = nn.Sequential(
            nn.Linear(2 * in_feature, in_feature),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(in_feature, class_num_sar)
        )
        self.SentimentClassifier = nn.Sequential(
            nn.Linear(2 * in_feature, in_feature),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(in_feature, class_num_sent)
        )

        self.apply(utils.init_weights)

    def forward(self, x, pooled):
        att_sar = self.sar_attention(x)
        att_sent = self.sent_attention(x)

        sar_x = att_sar.mul(torch.sigmoid(torch.matmul(att_sar, self.W_inter) + self.b_inter))
        sent_x = att_sent.mul(torch.sigmoid(torch.matmul(att_sent, self.W_inter) + self.b_inter))

        sar_xx = torch.cat([sar_x, pooled], 1)
        sent_xx = torch.cat([sent_x, pooled], 1)

        sar_out = self.sracasmClassifier(sar_xx)
        sent_out = self.SentimentClassifier(sent_xx)
        return sar_out, sent_out

class MTClassifier0(nn.Module):
    def __init__(self, in_feature, class_num_sar=1, class_num_sent=3 , dropout_prob=0.2):
        super(MTClassifier0, self).__init__()

        self.sracasmClassifier = nn.Linear(in_feature, class_num_sar)
        self.SentimentClassifier = nn.Linear(in_feature, class_num_sent)
        self.apply(utils.init_weights)

    def forward(self, x, pooled):
        sar_out = self.sracasmClassifier(pooled)
        sent_out = self.SentimentClassifier(pooled)
        return sar_out, sent_out

class MTClassifier4(nn.Module):
    def __init__(self, in_feature, class_num_sar=1, class_num_sent=3 , dropout_prob=0.2):
        super(MTClassifier4, self).__init__()
        self.W_inter = nn.Parameter(nn.init.xavier_normal_(torch.ones((in_feature, in_feature))))
        self.b_inter = nn.Parameter(torch.zeros(in_feature))
        self.sar_attention = AttentionWithContext(in_feature)
        self.sent_attention = AttentionWithContext(in_feature)
        self.auxTaskCls = nn.Linear(in_feature, 1)

        self.sracasmClassifier = nn.Sequential(
            nn.Linear(2 * in_feature, in_feature),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(in_feature, class_num_sar)
        )
        self.SentimentClassifier = nn.Sequential(
            nn.Linear(2 * in_feature, in_feature),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(in_feature, class_num_sent)
        )

        self.apply(utils.init_weights)

    def forward(self, x, pooled):
        att_sar = self.sar_attention(x)
        att_sent = self.sent_attention(x)

        att_all = torch.cat([att_sar, att_sent], 0)
        out_aux = self.auxTaskCls(att_all)

        sar_x = att_sar.mul(torch.sigmoid(torch.matmul(att_sar, self.W_inter) + self.b_inter))
        sent_x = att_sent.mul(torch.sigmoid(torch.matmul(att_sent, self.W_inter) + self.b_inter))


        sar_xx = torch.cat([sar_x, pooled], 1)
        sent_xx = torch.cat([sent_x, pooled], 1)

        sar_out = self.sracasmClassifier(sar_xx)
        sent_out = self.SentimentClassifier(sent_xx)
        return sar_out, sent_out, out_aux

class CLSClassifier(nn.Module):
    def __init__(self, in_feature, class_num=1, dropout_prob=0.2):
        super(CLSClassifier, self).__init__()
        self.attention = AttentionWithContext(in_feature)

        self.Classifier = nn.Sequential(
            nn.Linear(2 * in_feature, in_feature),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(in_feature, class_num)
        )

        self.apply(utils.init_weights)

    def forward(self, x, pooled):
        att = self.attention(x)

        xx = torch.cat([att, pooled], 1)

        out = self.Classifier(xx)
        return out

class CLSlayer(nn.Module):
    def __init__(self, in_feature, class_num=1):
        super(CLSlayer, self).__init__()

        self.Classifier = nn.Linear(in_feature, class_num)
        self.apply(utils.init_weights)

    def forward(self, x, pooled):

        out = self.Classifier(pooled)

        return out

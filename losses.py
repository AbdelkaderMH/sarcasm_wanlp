import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelMultitaskLoss(nn.Module):
    def __init__(self):
        super(ModelMultitaskLoss, self).__init__()
        self.eta = nn.Parameter(torch.tensor([0.0,0.0]))

    def forward(self, loss_1, loss_2,):
        total_loss_1 = loss_1 * torch.exp(-self.eta[0]) + self.eta[0]
        total_loss_2 = loss_2 * torch.exp(-self.eta[1]) + self.eta[1]

        total_loss = total_loss_1 + total_loss_2
        return total_loss


class F1_Loss(nn.Module):

    def __init__(self, epsilon=1e-7, num_class=3):
        super().__init__()
        self.epsilon = epsilon
        self.num_class = num_class
        self.ce = nn.CrossEntropyLoss().to(device)

    def forward(self, y_pred, y_true ):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        loss = self.ce(y_pred, y_true)
        y_true = F.one_hot(y_true, self.num_class).float()
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).float()
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).float()
        fp = ((1 - y_true) * y_pred).sum(dim=0).float()
        fn = (y_true * (1 - y_pred)).sum(dim=0).float()

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return loss - f1.mean()

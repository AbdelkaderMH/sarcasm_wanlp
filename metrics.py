from numpy import nanmean
import torch
def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum() / target.numel()
    return acc


class BinaryClassificationMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.pre = 0
        self.rec = 0
        self.f1 = 0

    def update(self, output, target):
        pred = output >= 0.5
        truth = target >= 0.5
        pred = pred.to(torch.float)
        truth = truth.to(torch.float)
        self.tp += pred.mul(truth).sum(0).float()
        self.tn += (1 - pred).mul(1 - truth).sum(0).float()
        self.fp += pred.mul(1 - truth).sum(0).float()
        self.fn += (1 - pred).mul(truth).sum(0).float()
        self.acc = (self.tp + self.tn).sum() / (self.tp + self.tn + self.fp + self.fn).sum()
        self.pre = self.tp / (self.tp + self.fp)
        self.rec = self.tp / (self.tp + self.fn)
        self.f1 = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)
        self.avg_acc = nanmean(self.acc.cpu())
        self.avg_pre = nanmean(self.pre.cpu())
        self.avg_rec = nanmean(self.rec.cpu())
        self.avg_f1 = nanmean(self.f1.cpu())


def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    pred = y.to(torch.float)
    truth = y_hat.to(torch.float)
    tp = pred.mul(truth).sum(0).float()
    fp = pred.mul(1 - truth).sum(0).float()
    fn = (1 - pred).mul(truth).sum(0).float()
    tn = (1 - pred).mul(1 - truth).sum(0).float()
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = cost.mean()  # average on all labels
    return macro_cost
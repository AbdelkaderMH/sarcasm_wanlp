import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.metrics import accuracy_score, f1_score



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

def f1_loss(preds, target):
    target = target
    predict = preds.squeeze()
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1 - f1.mean()

def accuracy(preds, y):
    all_output = preds.float().cpu()
    all_label = y.float().cpu()
    _, predict = torch.max(all_output, 1)
    acc = accuracy_score(all_label.numpy(), torch.squeeze(predict).float().numpy())
    return acc

def calc_accuracy(preds,y):
    predict = torch.argmax(preds, dim=1)
    accuracy = torch.sum(predict == y.squeeze()).float().item()
    return accuracy / float(preds.size()[0])

def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds).squeeze()

    correct = (rounded_preds == y).float()
    acc = correct.sum() / y.size(0)
    return acc



def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:


    y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return 1 - f1


def fscore(preds, y):
    all_output = preds.clone().detach().float().cpu()
    all_label = y.clone().detach().float().cpu()
    _, predict = torch.max(all_output, 1)
    acc = f1_score(all_label.numpy(), torch.squeeze(predict).float().numpy(), average='macro')
    return acc



def f1score(predict, target):
    target = target
    predict = predict.squeeze()
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return f1.mean()



def f1_loss(preds, target):
    target = target
    predict = preds.squeeze()
    #lack_cls = target.sum(dim=0) == 0
    #if lack_cls.any():
    #    loss += F.binary_cross_entropy_with_logits(
    #        predict[:, lack_cls], target[:, lack_cls])
    loss = F.binary_cross_entropy_with_logits(predict, target)
    predict = torch.sigmoid(predict)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1 - f1.mean() + loss #+ circle_loss(preds, target)

def fscore_loss(y_pred, target, beta=1, epsilon=1e-8):
    y_true = target.unsqueeze(1)
    TP = (y_pred * y_true).sum(dim=1)
    FP = ((1 - y_pred) * y_true).sum(dim=1)
    FN = (y_pred * (1 - y_true)).sum(dim=1)
    fbeta = (1 + beta ** 2) * TP / ((1 + beta ** 2) * TP + (beta ** 2) * FN + FP + epsilon)
    fbeta = fbeta.clamp(min=epsilon, max=1 - epsilon)
    return 1 - fbeta.mean()


def macro_double_soft_f1(y, y_hat):
    pred = y_hat.to(torch.float).unsqueeze(1)
    truth = y.to(torch.float).unsqueeze(1)
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

def mcc_loss(outputs_target, temperature=3, class_num=2):
    train_bs = outputs_target.size(0)
    outputs_target_temp = outputs_target / temperature
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    target_entropy_weight = Entropy(target_softmax_out_temp).detach()
    target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
    target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
    cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(target_softmax_out_temp)
    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
    return mcc_loss

def EntropyLoss(input_):
    # print("input_ shape", input_.shape)
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out + 1e-5)))
    return entropy / float(input_.size(0))

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
import os
from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import pdb
import collections


def one_hot(scores, labels):
    _labels = torch.zeros_like(scores)
    _labels.scatter_(dim=1, index=labels, value=1)#scatter_  带下划线的函数 是inplace  ，就是内部赋值，没有返回值
    _labels.requires_grad = False
    return _labels



def jaccard(intersection, union, eps=1e-15):
    return 1 - (intersection + eps) / (union - intersection + eps)

def dice(intersection, union, eps=1e-15):
    return 1 - (2. * intersection + eps) / (union + eps)

def dice2(intersection, union, eps=1e-15):
#*F.sigmoid(union - 2.*intersection)
    return 1 - (2. * intersection + eps) / (20.*(union - 2.*intersection) + 2. * intersection + eps)

class BCESoftJaccardDice(nn.Module):

    def __init__(self, bce_weight=0., mode="dice", eps=1e-15, weight=None):
        super(BCESoftJaccardDice, self).__init__()
        self.nll_loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        self.bce_weight = bce_weight
        self.eps = eps
        self.mode = mode

    def forward(self, outputs, targets):

        targets = targets.unsqueeze(1)
        targets = one_hot(outputs, targets)
        loss = self.bce_weight * self.nll_loss(outputs, targets)

        if self.bce_weight < 1.:
            targets = (targets == 1).float()
            outputs = torch.sigmoid(outputs)
            intersection = (outputs * targets).sum()
            union = outputs.sum() + targets.sum()
            if self.mode == "dice":
                score = dice(intersection, union, self.eps)
            elif self.mode == "jaccard":
                score = jaccard(intersection, union, self.eps)
            loss += (1 - self.bce_weight) * (score)
        return loss

class BCESoftJaccardDiceRateChange(nn.Module):

    def __init__(self, mode="dice", eps=1e-15, weight=None):
        super(BCESoftJaccardDiceRateChange, self).__init__()
        self.nll_loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        self.eps = eps
        self.mode = mode


    def forward(self, outputs, targets):
        targets = targets.unsqueeze(1)
        targets = one_hot(outputs, targets)
       
        # targets = (targets == 1).float()
        outputs = torch.sigmoid(outputs)
        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum()

        loss = dice2(intersection, union, self.eps)

        return loss

def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

 
def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    _,pred = y_pred.topk(1, 1, True, True)
    batch_size = y_actual.size(0)
 
    pred = pred.squeeze()
    correct = len(pred[pred == y_actual])
    wrong = len(pred[pred != y_actual])
    return correct/(correct+wrong)

    return res

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):

        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs,dim=1)) ** self.gamma * F.log_softmax(inputs,dim=1), targets)



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightDiceLoss(nn.Module):
    def __init__(self):
        super(WeightDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        num_sum = torch.sum(targets, dim=(0, 2, 3, 4))
        w = torch.Tensor([0, 0, 0, 0]).cuda()
        for i in range(targets.size(1)):
            if (num_sum[i] < 1):
                w[i] = 0
            else:
                w[i] = (0.1 * num_sum[i] + num_sum[i - 1] + num_sum[i - 2] + 1) / (torch.sum(num_sum) + 1)
        # print(w)
        inter = w * torch.sum(targets * logits, dim=(0, 2, 3, 4))
        inter = torch.sum(inter)

        union = w * torch.sum(targets + logits, dim=(0, 2, 3, 4))
        union = torch.sum(union)
        return 1 - 2. * (inter+smooth) / (union+smooth)

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class MixLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x, y):
        lf, lfw = [], []
        for i, v in enumerate(self.args):
            if i % 2 == 0:
                lf.append(v)
            else:
                lfw.append(v)
        mx = sum([w * l(x, y) for l, w in zip(lf, lfw)])
        return mx


class DiceLoss(nn.Module):
    def __init__(self, image=False):
        super().__init__()
        self.image = image

    def forward(self, x, y):
        x = x.sigmoid()
        i, u = [t.flatten(1).sum(1) if self.image else t.sum() for t in [x * y, x + y]]

        dc = (2 * i + 1) / (u + 1)
        dc = 1 - dc.mean()
        return dc


class GHMCLoss(nn.Module):
    def __init__(self, mmt=0, bins=10):
        super().__init__()
        self.mmt = mmt
        self.bins = bins
        self.edges = [x / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6

        if mmt > 0:
            self.acc_sum = [0] * bins

    def forward(self, x, y):
        w = torch.zeros_like(x)
        g = torch.abs(x.detach().sigmoid() - y)

        n = 0
        t = reduce(lambda x, y: x * y, w.shape)
        for i in range(self.bins):
            ix = (g >= self.edges[i]) & (g < self.edges[i + 1]); nb = ix.sum()
            if nb > 0:
                if self.mmt > 0:
                    self.acc_sum[i] = self.mmt * self.acc_sum[i] + (1 - self.mmt) * nb
                    w[ix] = t / self.acc_sum[i]
                else:
                    w[ix] = t / nb
                n += 1
        if n > 0:
            w = w / n

        gc = F.binary_cross_entropy_with_logits(x, y, w, reduction='sum') / t
        return gc


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        ce = F.binary_cross_entropy_with_logits(x, y)
        fc = self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce
        return fc
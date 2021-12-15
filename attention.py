# -*- coding: utf-8 -*-
# !@time: 2021/12/13 下午9:56
# !@author: superMC @email: 18758266469@163.com
# !@fileName: selayer.py
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, c = x.size()
        y = self.avg_pool(x.transpose(1, 2).contiguous()).view(b, c)
        y = self.fc(y).view(b, 1, c)
        return x * y.expand_as(x)


class TimeAttention(nn.Module):
    def __init__(self, reduction=16):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        b, t, c = x.size()
        y = self.avg_pool(x).view(b, t)
        y = F.softmax(y, dim=1).view(b, t, 1)
        return x * y.expand_as(x)

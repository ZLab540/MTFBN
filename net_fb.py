# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual_dense_block import RDB
from utils import *

BN_MOMENTUM = 0.1
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(channel_in, channel_out, stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out

class net_fb(nn.Module):
    def __init__(self,CH=32):
        super(net_fb, self).__init__()
        self.conv00 = BasicBlock(6,32)
        self.conv01 = RDB(32,4,16)
        self.conv02 = RDB(32,4,16)
        self.conv03 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x0,y0,out_all = False):
        x1 = self.conv00(torch.cat([x0,y0],1))
        x1 = self.conv01(x1)
        x1 = self.conv02(x1)
        x2 = self.conv03(x1)
        x3 = self.conv13(x1)

        return torch.cat([x2,x3],1)
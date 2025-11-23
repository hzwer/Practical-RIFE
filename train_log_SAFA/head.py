import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.warplayer import *

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.PReLU(out_planes)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std).to(device)
        self.weight.data = torch.eye(c).view(c, c, 1, 1).to(device)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean).to(device)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean).to(device)
        self.requires_grad = False

class Head(nn.Module):
    def __init__(self, c):
        super(Head, self).__init__()
        model = models.resnet18(pretrained=False)
        self.cnn0 = nn.Sequential(*nn.ModuleList(model.children())[:3])
        self.cnn1 = nn.Sequential(
            *list(model.children())[3:5],
        )
        self.cnn2 = nn.Sequential(
            *list(model.children())[5:6],
        )
        self.out0 = nn.Conv2d(64, c, 1, 1, 0)
        self.out1 = nn.Conv2d(64, c, 1, 1, 0)
        self.out2 = nn.Conv2d(128, c, 1, 1, 0)
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)                        
    def forward(self, x):
        x = self.normalize(x)
        f0 = self.cnn0(x)
        f1 = self.cnn1(f0)
        f2 = self.cnn2(f1)
        f0 = self.out0(f0)
        f1 = F.interpolate(self.out1(f1), scale_factor=2.0, mode="bilinear")
        f2 = F.interpolate(self.out2(f2), scale_factor=4.0, mode="bilinear")
        return f0 + f1 + f2

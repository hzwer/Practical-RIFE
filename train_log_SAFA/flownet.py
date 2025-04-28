import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from train_log.warplayer import warp
from train_log.head import Head

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True, groups=groups),        
        nn.LeakyReLU(0.2, True)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True)
    )

class Resblock(nn.Module):
    def __init__(self, c, dilation=1):
        super(Resblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, 2*c, 3, 2, dilation, dilation=dilation, groups=1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(2*c, c, 4, 2, 1)
        )
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.prelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        y = self.conv(x)
        return self.prelu(y * self.beta + x)

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.bernoulli(x)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None
    
class RecurrentBlock(nn.Module):
    def __init__(self, c, dilation=1, depth=6):
        super(RecurrentBlock, self).__init__()
        self.conv_stem = conv(3*c+6+1, c, 3, 1, 1, groups=1)
        self.conv_backbone = torch.nn.ModuleList([])        
        self.depth = depth
        for i in range(depth):
            self.conv_backbone.append(Resblock(c, dilation))
        
    def forward(self, x, i0, i1, flow, timestep, convflow, getscale):
        flow_down = F.interpolate(flow, scale_factor=0.5, mode="bilinear")
        i0 = warp(i0, flow_down[:, :2] * 0.5)
        i1 = warp(i1, flow_down[:, 2:4] * 0.5)
        x = torch.cat((x, flow_down, i0, i1, timestep), 1)
        scale = RoundSTE.apply(getscale(x)).unsqueeze(2).unsqueeze(3)
        feat = 0
        if scale.shape[0] != 1 or (scale[:, 0:1].mean() > 0.5 and scale[:, 1:2].mean() > 0.5):
            x0 = self.conv_stem(x)
            for i in range(self.depth):
                x0 = self.conv_backbone[i](x0)
            feat = feat + x0 * scale[:, 0:1] * scale[:, 1:2] 

        if scale.shape[0] != 1 or (scale[:, 0:1].mean() < 0.5 and scale[:, 1:2].mean() > 0.5):
            x1 = self.conv_stem(F.interpolate(x, scale_factor=0.5, mode="bilinear"))
            for i in range(self.depth):
                x1 = self.conv_backbone[i](x1)
            feat = feat + F.interpolate(x1, scale_factor=2.0, mode="bilinear") * (1 - scale[:, 0:1]) * scale[:, 1:2]

        if scale.shape[0] != 1 or scale[:, 1:2].mean() < 0.5:
            x2 = self.conv_stem(F.interpolate(x, scale_factor=0.25, mode="bilinear"))
            for i in range(self.depth):
                x2 = self.conv_backbone[i](x2)
            feat = feat + F.interpolate(x2, scale_factor=4.0, mode="bilinear") * (1 - scale[:, 1:2])
        return feat, convflow(feat) + flow, i0, i1, scale

class Flownet(nn.Module):
    def __init__(self, block_num, c=64):
        super(Flownet, self).__init__()
        self.convimg = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, c, 3, 1, 1),
        )
        self.convblock = torch.nn.ModuleList([])
        self.block_num = block_num
        self.convflow = nn.Sequential(
            nn.Conv2d(c, 4*6, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.getscale = nn.Sequential(
            conv(3*c+6+1, c, 1, 1, 0),
            conv(c, c, 1, 2, 0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c, 2),
            nn.Sigmoid()
        )
        for i in range(self.block_num):
            self.convblock.append(RecurrentBlock(c, 1, 2))

    def extract_feat(self, x):
        i0 = self.convimg(x[:, :3])
        i1 = self.convimg(x[:, 3:6])
        return i0, i1
        
    def forward(self, i0, i1, feat, timestep, flow):
        flow_list = []
        feat_list = []
        scale_list = []
        for i in range(self.block_num):
            feat, flow, w0, w1, scale = self.convblock[i](feat, i0, i1, flow, timestep, self.convflow, self.getscale)
            flow_list.append(flow)
            feat_list.append(feat)
            scale_list.append(scale)
        return flow_list, feat_list, torch.cat(scale_list, 1)
        
class SAFA(nn.Module):
    def __init__(self):
        super(SAFA, self).__init__()
        c=96
        self.block = Flownet(4, c=c)
        self.shuffle = conv(2*c, c, 3, 1, 1, groups=1)
        self.lastconv0 = nn.Sequential(
            conv(4*c, c, 3, 1, 1),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
        )
        self.lastconv1 = nn.Sequential(
            conv(5*c, 2*c, 3, 1, 1),
            nn.Conv2d(2*c, 12, 3, 1, 1),
            nn.PixelShuffle(2),
        )
        
    def inference(self, lowres, timestep=None):
        merged = []
        i0, i1 = self.block.extract_feat(lowres)
        timestep = (lowres[:, :1] * 0).detach()
        timestep = F.interpolate(timestep, scale_factor=0.5, mode="bilinear")
        for i in range(2):
            if i == 1:
                tmp = i0
                i0 = i1
                i1 = tmp
            feat = self.shuffle(torch.cat((i0, i1), 1))
            flow_list, feat_list, soft_scale = self.block(i0, i1, feat, timestep, (lowres[:, :6] * 0).detach())
            flow_sum = flow_list[-1]
            flow_down = F.interpolate(flow_sum, scale_factor=0.5, mode="bilinear")
            w1 = warp(i1, flow_down[:, 2:4] * 0.5)
            lastfeat = torch.cat((i0, w1, feat_list[-1], feat_list[-3]), 1)
            res = self.lastconv1(torch.cat((self.lastconv0(lastfeat), lastfeat), 1))
            merged.append(torch.clamp(res, 0, 1))
        return merged

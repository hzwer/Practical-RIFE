import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from train_log.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from train_log.flownet import *
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Model:
    def __init__(self, local_rank=-1):
        self.flownet = SAFA()
        self.optimG = AdamW(self.flownet.parameters())
        self.device()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            
    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def inference(self, i0, i1, timestep):
        return self.flownet.inference(torch.cat((i0, i1), 1), timestep)
        
    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if device == torch.device('cpu'):
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location=torch.device('cpu'))))
        elif device == torch.device('mps'):
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location=torch.device('mps'))))
        elif rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def update(self, imgs, lowres, learning_rate=0, timestep=0.5, mul=1, training=True):
        for param_group in self.optimG.param_groups:
            if param_group['name'] == 'flow':
                param_group['lr'] = learning_rate 
            else:
                param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, -3:]
        if training:
            self.train()
            for m in self.flownet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()                 
        else:
            self.eval()
        flow, scale, merged = self.flownet(lowres, timestep=timestep, training=training)
        loss_l1 = 0
        for i in range(3):
            loss_l1 += (imgs[:, i*3:i*3+3] - merged[i]).abs().mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.flownet.parameters(), 1.0)
            self.optimG.step()
        return merged, {
            'scale': scale,
            'flow': flow[:, :2],
            'loss_l1': loss_l1,
            }

import torch
import torch.nn as nn
import torch.autograd as autograd
from torchvision import models
import torch.nn.functional as F
import numpy as np
import math

import torch
import torch.nn as nn

"""
    Mapping network
    Input: two tensor of size (batchsize, 512, 4, 4)
    Output: a tensor of size (batchsize, 480)

    how to combine two tensor into one tensor is a challenge.
"""
class MappingResBlock(nn.Module):
    def __init__(self, in_channels, ksize=3, padding=0, stride=1):
        super(MappingResBlock, self).__init__()
        # Initialize the conv scheme
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, ksize, padding=padding, stride=stride),
            #nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace = False),
            nn.Conv2d(in_channels, in_channels, ksize, padding=padding, stride=stride)
            #nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = 0.1 * out + residual
        return out

class MappingNet(nn.Module):
    def __init__(self, in_channels, out_channels, out_num, input_norm=False, output_norm=True):
        super(MappingNet, self).__init__()
        self.input_norm = input_norm
        self.output_norm = False
        self.out_num = out_num
        self.outchannel = out_channels
        # Head block
        self.head = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels*2, 3, stride=2, padding=1), #in: 2048,4,4  out: 1024,3,3
            nn.LeakyReLU(0.2, inplace = False),
            nn.Conv2d(in_channels*2, in_channels, 2, stride=1, padding=0) #in:1024,3,3 out:512,1,1
        )
        # Bottle neck  感觉5个resblock应该够了把...
        self.bottle = nn.Sequential(
            MappingResBlock(in_channels, 1, 0, 1),
            MappingResBlock(in_channels, 1, 0, 1),
            MappingResBlock(in_channels, 1, 0, 1),
            MappingResBlock(in_channels, 1, 0, 1),
            MappingResBlock(in_channels, 1, 0, 1),
        #    MappingResBlock(in_channels),
        #    MappingResBlock(in_channels),
        #    MappingResBlock(in_channels)
        )
        self.final = nn.Linear(in_channels, out_channels * out_num) #in_channels=512, out_channels = 480

    def forward(self, x_father, x_mother):
        assert x_father.shape==x_mother.shape, 'shape of x_father and x_mother is different, x_father:{} x_mother'.format(x_father.shape, x_mother.shape)
        if self.input_norm:
            x_father = (x_father - x_father.mean(dim=[1,2,3]).reshape(x_father.shape[0],1,1,1)) / x_father.var(dim=[1,2,3]).reshape(x_father.shape[0],1,1,1)
            x_mother = (x_mother - x_mother.mean(dim=[1,2,3]).reshape(x_mother.shape[0],1,1,1)) / x_mother.var(dim=[1,2,3]).reshape(x_mother.shape[0],1,1,1)
        x   = torch.cat((x_father, x_mother), dim=1)  #在channel维进行合并 -> [bs, 1024, 4, 4]
        #head block        
        out = self.head(x)
        # Bottle neck
        out = self.bottle(out)
        # Final conv
        out = out.reshape(out.shape[0], out.shape[1])
        out = self.final(out)

        if self.output_norm:
            for i in range(self.out_num):
                #out = (out - out.mean(dim=1).reshape(out.shape[0], 1)) / out.var(dim=1).reshape(out.shape[0], 1)
                out[:, (self.outchannel * i):(self.outchannel * (i + 1))] = \
                    (out[:, (self.outchannel * i):(self.outchannel * (i + 1))] - out[:, (self.outchannel * i):(self.outchannel * (i + 1))].mean(dim=1).reshape(out[:, (self.outchannel * i):(self.outchannel * (i + 1))].shape[0], 1)) / out[:, (self.outchannel * i):(self.outchannel * (i + 1))].var(dim=1).reshape(out[:, (self.outchannel * i):(self.outchannel * (i + 1))].shape[0], 1)
        out = out.reshape(out.shape[0], self.out_num, -1)
        out = [out[:,i,:] for i in range(self.out_num)]
        return out #[batchsize, 512]


if __name__ == '__main__':
    x_father = torch.randn((1,1024,4,4)).cuda()
    x_mother = torch.randn((1,1024,4,4)).cuda()

    net = MappingNet(512, 480, 5).cuda()

    code_of_child = net(x_father, x_mother)
    print(code_of_child.shape)

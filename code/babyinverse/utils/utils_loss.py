import math
import random

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        # conv1_2
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # conv2_2
        for x in range(3, 8):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # conv3_2
        for x in range(8, 13):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # conv4_2
        for x in range(13, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # conv5_1
        for x in range(23, 31):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, X):
        # if down:
        #    X = self.down(X)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]  #return 5 fmap
        return out

# define vgg19 loss
class VGGLoss(nn.Module):
    def __init__(self, weight_vgg=10., norm_input=False):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
#        self.criterion = nn.L1Loss().to(device)

        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.weight_vgg = weight_vgg

        # add normalization
        self.norm = norm_input
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def __normalization(self, img, mean, std):
        mean = torch.tensor(mean).view(-1, 1, 1).to(img.device)
        std = torch.tensor(std).view(-1, 1, 1).to(img.device)
        return (img - mean) / std

    def _compute_vgg(self, x_vgg, y_vgg):
        loss_vgg = 0.
        for i in range(len(x_vgg)):
            loss_vgg += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss_vgg

    def forward(self, x, y):
        # with torch.no_grad():
        # when batch = 1 try 1024 resolution
        if x.size(2) > 512:
            x = F.interpolate(x, scale_factor=1 / 2, mode='area')
            y = F.interpolate(y, scale_factor=1 / 2, mode='area')
        if self.norm:
            x, y = self.__normalization((x+1)/2, self.mean, self.std), self.__normalization((y+1)/2, self.mean, self.std)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss_vgg = 0.

        if self.weight_vgg > 0:
            # loss_vgg = self._compute_vgg(x_vgg, y_vgg) + 2. * identity_loss(x_vgg[-1], y_vgg[-1])
            loss_vgg = self._compute_vgg(x_vgg, y_vgg)

        loss = self.weight_vgg * loss_vgg
        return loss
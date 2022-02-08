import torch
import torch.nn as nn
import torch.autograd as autograd
from torchvision import models
import torch.nn.functional as F
import numpy as np
import math


# vgg encoder
class Vgg16(torch.nn.Module):
    def __init__(self, 
                pre_train=True, 
                requires_grad=False, 
                vae_encoder=False, 
                global_pooling=True,
                if_downsample=False):
        super(Vgg16, self).__init__()

        self.use_vae = vae_encoder
        self.global_pooling = global_pooling
        self.if_downsample = if_downsample

        vgg_pretrained_features = models.vgg16(pretrained=pre_train).features
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
        for x in range(8, 15):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # conv4_2
        for x in range(15, 22):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # conv5_1
        for x in range(22, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # global avg pooling
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.down = nn.AvgPool2d(3, 2, 1)

        # fc layer for encoding to lantent B * 512
        if self.use_vae:
            self.fc_mu = nn.Linear(512, 512)
            self.fc_var = nn.Linear(512, 512)
        else:
            self.fc_layer = nn.Sequential(nn.Linear(512*4*4, 480*1))


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, X):
        if self.if_downsample:
           X = self.down(X)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        # out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        if self.global_pooling:
            h_ = self.global_pool(h_relu5)

        h_ = h_.view(h_.size(0), -1)

        if self.use_vae:
            mu, var = self.fc_mu(h_), self.fc_var(h_)
            z = self.reparameterize(mu, var)
            return z, mu, var
        else:
            z = self.fc_layer(h_)
            return z.view(-1, 480), [h_relu5, h_relu4, h_relu3]



#------------------------------------Dis_net-----------------------#
class BlurLayer(nn.Module):
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=False,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(BlurLayer(blur_kernel))
            # layers.append(DownSampleLayer(scale=1/2., pad=(pad0, pad1)))
            # layers.append(nn.AvgPool2d(3, 2, 1))

            stride = 2
            if kernel_size == 3:
                self.padding = 1
            else:
                self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], use_sn=False):
        super().__init__()

        if use_sn:
            self.conv1 = nn.utils.spectral_norm(ConvLayer(in_channel, in_channel, 3))
            self.conv2 = nn.utils.spectral_norm(ConvLayer(in_channel, out_channel, 3, downsample=True))
            self.skip = nn.utils.spectral_norm(ConvLayer(
                in_channel, out_channel, 1, downsample=True, activate=False, bias=False
            ))
        else:
            self.conv1 = ConvLayer(in_channel, in_channel, 3)
            self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=True, activate=False, bias=False
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation
        if activation == "scaled_leakyrelu":
            self.active_layer = ScaledLeakyReLU()

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = self.active_layer(out)
            # out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class Discriminator(nn.Module):
    def __init__(self, size, input_channel=3, ndf=16, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], use_sigmoid=True,
                 use_sn=False):
        super().__init__()

        channels = {
            2: ndf * 32,
            4: ndf * 32,
            8: ndf * 32,
            16: ndf * 32,
            32: ndf * 32,
            64: ndf * 16 * channel_multiplier,
            128: ndf * 8 * channel_multiplier,
            256: ndf * 4 * channel_multiplier,
            512: ndf * 2 * channel_multiplier,
            1024: ndf * channel_multiplier,
        }

        convs = [ConvLayer(input_channel, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        # if use_sn:
        #     self.final_conv = nn.utils.spectral_norm(nn.Conv2d(in_channel + 1, channels[4], kernel_size=3))
        #     self.final_linear = nn.Sequential(
        #         nn.utils.spectral_norm(nn.Linear(channels[4] * 4 * 4, channels[4])),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #         nn.utils.spectral_norm(nn.Linear(channels[4], 1)),
        #     )
        # else:
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            # EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[2] * 2 * 2, channels[2], activation='scaled_leakyrelu'),
            EqualLinear(channels[2], 1),
        )
        self.use_sigmoid = use_sigmoid

    def forward(self, input):
        input = F.interpolate(input, scale_factor=0.5)
        if isinstance(input, list):
            input = torch.cat(input, dim=1)
        out = self.convs(input)


        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        if self.use_sigmoid:
            out = F.sigmoid(out)
        return out

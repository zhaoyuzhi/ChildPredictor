import torch
import torch.nn as nn
from Models.network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                Generator
# ----------------------------------------
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # Encoder
        self.Enc = nn.Sequential(
            Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = opt.activ_g),
            Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_g, activation = opt.activ_g),
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_g, activation = opt.activ_g),
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_g, activation = opt.activ_g),
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_g, activation = opt.activ_g),
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_g, activation = opt.activ_g)
        )
        # Decoder
        self.Dec = nn.Sequential(
            TransposeConv2dLayer(opt.start_channels * 4 + opt.attr_channels, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm_g, activation = opt.activ_g),
            TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm_g, activation = opt.activ_g),
            TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm_g, activation = opt.activ_g),
            TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm_g, activation = opt.activ_g),
            TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm_g, activation = opt.activ_g),
            Conv2dLayer(opt.start_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'tanh')
        )

    def enc(self, x):
        x = self.Enc(x)                                             # out: batch * 256 * 4 * 4
        return x

    def dec(self, x, attr):
        # Concatenate attribute
        attr_expand = attr.view(attr.shape[0], attr.shape[1], 1, 1).expand(attr.shape[0], attr.shape[1], x.shape[2], x.shape[3])
        x_con = torch.cat((x, attr_expand), 1)                      # out: batch * (256 + z_dim) * 4 * 4
        x = self.Dec(x_con)                                         # out: batch * 3 * 128 * 128
        return x

    def forward(self, x, attr, attr_modified):                      # for training
        # Encode
        x = self.enc(x)                                             # out: batch * 256 * 4 * 4
        # Decode
        img_recon = self.dec(x, attr)                               # out: batch * 3 * 128 * 128
        img_fake = self.dec(x, attr_modified)                       # out: batch * 3 * 128 * 128
        return img_recon, img_fake

    def val(self, x, attr):                                         # for validation
        # Encode
        x = self.enc(x)                                             # out: batch * 256 * 4 * 4
        # Decode
        x = self.dec(x, attr)                                       # out: batch * 3 * 128 * 128
        return x

# ----------------------------------------
#               Discriminator
# ----------------------------------------
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        # Down sampling
        self.block = nn.Sequential(
            Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = opt.activ_d, sn = True),
            Conv2dLayer(opt.start_channels, opt.start_channels, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_d, activation = opt.activ_d, sn = True),
            Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_d, activation = opt.activ_d, sn = True),
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_d, activation = opt.activ_d, sn = True),
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_d, activation = opt.activ_d, sn = True),
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_d, activation = opt.activ_d, sn = True),
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_d, activation = opt.activ_d, sn = True),
            Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = 'none', activation = opt.activ_d, sn = True)
        )
        # Final output
        self.fc_adv = nn.Sequential(
            nn.Linear(opt.start_channels * 8, opt.start_channels * 8),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(opt.start_channels * 8, 1)
        )
        self.fc_class = nn.Sequential(
            nn.Linear(opt.start_channels * 8, opt.start_channels * 8),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(opt.start_channels * 8, opt.attr_channels)
        )

    def forward(self, x):
        x = self.block(x)                                           # out: batch * 512 * 1 * 1
        x = x.view(x.size(0), -1)                                   # out: batch * 512
        out_adv = self.fc_adv(x)                                    # out: batch * 1
        out_class = self.fc_class(x)                                # out: batch * 4
        return out_adv, out_class

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm_g', type = str, default = 'bn', help = 'normalization type of networks')
    parser.add_argument('--norm_d', type = str, default = 'in', help = 'normalization type of networks')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'in channels for the main stream of generator')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'out channels for the main stream of generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--attr_channels', type = int, default = 4, help = 'noise channels')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    opt = parser.parse_args()
    print(opt)

    gen = Generator(opt).cuda()
    dis = Discriminator(opt).cuda()
    img = torch.randn(1, 3, 128, 128).cuda()
    attr = torch.randn(1, 4).cuda()

    img_recon, img_fake = gen(img, attr, attr)
    print(img_recon.shape, img_fake.shape)

    out_adv, out_class = dis(img)
    print(out_adv.shape, out_class.shape)

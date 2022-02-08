import torch
import torch.nn as nn



class VAENET(nn.Module):
    def __init__(self, args):
        super(VAENET, self).__init__()
        self.args = args
        self.in_channel = args.in_channel
        self.out_channel = args.out_channel
        self.max_channel = args.out_channel
        self.latent_dim = args.latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channel, 32, kernel_size=3, stride=2, padding=1),   #bx6x128x128   ->    bx32x64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),                  #bx32x64x64   ->    bx64x32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),                 #bx64x32x32   ->    bx128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),                #bx128x16x16   ->    bx256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),                #bx256x8x8   ->    bx256x4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, self.out_channel, kernel_size=3, stride=1, padding=1),                #bx256x4x4   ->    bx256x4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu = nn.Linear(self.out_channel*4*4, self.latent_dim)
        self.fc_logvar = nn.Linear(self.out_channel*4*4, self.latent_dim)
        self.fc_decoder = nn.Linear(self.latent_dim+8, self.out_channel*4*4)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.out_channel, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 6, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def reparameterize(self, mu, logvar):
        eps = torch.randn((mu.size(0), mu.size(1))).cuda()
        z = mu + eps*torch.exp(logvar/2)
        return z

    def forward(self, data, label_father, label_mother):
        code = self.encoder(data)                                #in: Bx6x128x128     out:Bx256x4x4
        mu = self.fc_mu(code.view(code.size(0), -1))             #in: Bx256*4*4     out:Bxlatent_dim
        logvar = self.fc_logvar(code.view(code.size(0), -1))     #in: Bx256*4*4     out:Bxlatent_dim
        z = self.reparameterize(mu, logvar)                      #in: Bxlatent_dim     out:Bxlatent_dim
        z = torch.cat((z, label_father, label_mother), dim=1)    #in: Bxlatent_dim     out:Bx(latent_dim+8)
        in_decoder = self.fc_decoder(z)                          #in: Bx(latent_dim+8)     out:Bx256*4*4
        out = self.decoder(in_decoder.view(in_decoder.size(0), self.out_channel, 4, 4))   #in: Bx256x4x4    out:Bx6*128*128

        return out, mu, logvar

# TODO: perform any necessary preprocessing on x (like scale pixel values to [-1, 1], etc)

import torch
import torch.nn as nn
import torch.nn.functional as F
import opts

# Pytorch uses [N, C, H, W]!
class Generator(nn.Module):
    def __init__(self, embed_size, num_classes):
        super(Generator, self).__init__()

        if not opts.USE_CLASSES:
            num_classes = 0
            
        self.main = nn.Sequential(
            nn.ConvTranspose2d(embed_size + num_classes, opts.NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opts.NGF * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(opts.NGF * 8, opts.NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opts.NGF * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(opts.NGF * 4, opts.NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opts.NGF * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(opts.NGF * 2, opts.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opts.NGF),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(opts.NGF, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    # z is (m, 1024), y is (m, n_classes)
    def forward(self, z, y):
        if opts.USE_CLASSES:
            zy = torch.cat((z, y), -1)
            return self.main(zy[:, :, None, None])
        return self.main(z[:, :, None, None])
        
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        if not opts.USE_CLASSES:
            num_classes = 0
        
        self.main = nn.Sequential(
            nn.Conv2d(3, opts.NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opts.NDF, opts.NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opts.NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opts.NDF * 2, opts.NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opts.NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opts.NDF * 4, opts.NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opts.NDF * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layers
        self.linear = nn.Linear(4 * 4 * opts.NDF * 8 + num_classes, 1, bias=False)

    # x is (m, 128, 128, 3), y is (m, n_classes)
    def forward(self, x, y):
        z = self.main(x).view(-1, 4 * 4 * opts.NDF * 8)
        if opts.USE_CLASSES:
            zy = torch.cat((z, y), -1)
            return torch.sigmoid(self.linear(zy))
        return torch.sigmoid(self.linear(z))

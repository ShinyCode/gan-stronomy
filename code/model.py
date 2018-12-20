# -------------------------------------------------------------
# proj:    gan-stronomy
# file:    model.py
# authors: Mark Sabini, Zahra Abdullah, Darrith Phan
# desc:    Conditional DCGAN architecture used for generative cooking.
# -------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import opts

class Generator(nn.Module):
    def __init__(self, latent_size, embed_size):
        super(Generator, self).__init__()

        if not opts.CONDITIONAL:
            print('WARNING: USING UNCONDITIONED GENERATOR')
            embed_size = 0
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size + embed_size, opts.NGF * 8, 4, 1, 0, bias=False),
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
        
    # z is [B, opts.LATENT_SIZE], emb is size [B, opts.EMBED_SIZE]
    def forward(self, z, emb):
        if not opts.CONDITIONAL:
            return self.main(z[:, :, None, None])
        zemb = torch.cat((z, emb), -1)
        return self.main(zemb[:, :, None, None])

class Discriminator(nn.Module):
    def __init__(self, embed_size):
        super(Discriminator, self).__init__()

        if not opts.CONDITIONAL:
            print('WARNING: USING UNCONDITIONED DISCRIMINATOR')
            embed_size = 0
        
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

        self.linear = nn.Linear(4 * 4 * opts.NDF * 8 + embed_size, 1, bias=False)

    # x is [B, 3, opts.IMAGE_SIZE, opts.IMAGE_SIZE], emb is size [B, opts.EMBED_SIZE]
    def forward(self, x, emb):
        z = self.main(x).view(-1, 4 * 4 * opts.NDF * 8)
        if not opts.CONDITIONAL:
            return torch.sigmoid(self.linear(z))
        zemb = torch.cat((z, emb), -1)
        return torch.sigmoid(self.linear(zemb))

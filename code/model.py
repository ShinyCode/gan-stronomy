# TODO: perform any necessary preprocessing on x (like scale pixel values to [-1, 1], etc)

import torch
import torch.nn as nn
import torch.nn.functional as F

def padding(f):
    return (f - 1) // 2

def output_padding(n_in, s, f):
    p = padding(f)
    return s * n_in - (n_in - 1) * s + 2 * p - f

def Deconv2dSame(input_size, in_channels, out_channels, kernel_size, stride=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding(kernel_size), output_padding=output_padding(input_size, stride, kernel_size))

def Conv2dSame(in_channels, out_channels, kernel_size, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding(kernel_size))


# Pytorch uses [N, C, H, W]!
class Generator(nn.Module):
    def __init__(self, embed_size, num_classes):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(embed_size + num_classes, 1024, 4, 1, 0),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )
        
    # z is (m, 1024), y is (m, n_classes)
    def forward(self, z, y):
        zy = torch.cat((z, y), -1)
        return self.main(zy[:, :, None, None])

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.ELU(inplace=True)
        )
        # Layers
        self.linear = nn.Linear(16384 + num_classes, 1)

    # x is (m, 128, 128, 3), y is (m, n_classes)
    def forward(self, x, y):
        z = self.main(x).view(-1, 16384)
        print(z.shape)
        
        zy = torch.cat((z, y), -1)
        return torch.sigmoid(self.linear(zy))

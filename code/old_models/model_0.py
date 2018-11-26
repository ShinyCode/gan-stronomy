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
        # TODO: Add padding for nn.ConvTranspose2d (as if we were using padding "same")
        self.linear1 = nn.Linear(embed_size + num_classes, 1024)
        self.linear2 = nn.Linear(1024, 1024) # Restriction: nn.Linear(b, a) -> nn.Linear(c, b)
        self.deconv2d1 = Deconv2dSame(4, 64, 32, 5, stride=2) # (64, 4, 4) -> (32, 8, 8)
        self.deconv2d2 = Deconv2dSame(8, 32, 32, 5, stride=2) # -> (32, 16, 16)
        self.deconv2d3 = Deconv2dSame(16, 32, 32, 5, stride=2) # -> (32, 32, 32)
        self.deconv2d4 = Deconv2dSame(32, 32, 3, 5, stride=2) # -> (3, 64, 64)

    # x is (m, 1024), y is (m, n_classes)
    def forward(self, z, y):
        zy = torch.cat((z, y), -1)
        linear_a1 = F.elu(self.linear1(zy))
        linear_a2 = F.elu(self.linear2(linear_a1))
        linear_a2_reshape = linear_a2.view(-1, 64, 4, 4)
        deconv_a1 = F.elu(self.deconv2d1(linear_a2_reshape))
        deconv_a2 = F.elu(self.deconv2d2(deconv_a1))
        deconv_a3 = F.elu(self.deconv2d3(deconv_a2))
        deconv_a4 = torch.tanh(self.deconv2d4(deconv_a3))
        return deconv_a4

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        # Layers
        self.conv2d1 = Conv2dSame(3, 32, 5, stride=2) # (3, 64, 64) -> (32, 32, 32)
        self.conv2d2 = Conv2dSame(32, 32, 5, stride=2) # -> (32, 16, 16)
        self.conv2d3 = Conv2dSame(32, 32, 5, stride=2) # -> (32, 8, 8)
        self.conv2d4 = Conv2dSame(32, 64, 5, stride=2) # -> (64, 4, 4)
        self.linear1 = nn.Linear(1024 + num_classes, 512)
        self.linear2 = nn.Linear(512, 1)

    # x is (m, 128, 128, 3), y is (m, n_classes)
    def forward(self, x, y):
        conv_a1 = F.elu(self.conv2d1(x))
        conv_a2 = F.elu(self.conv2d2(conv_a1))
        conv_a3 = F.elu(self.conv2d3(conv_a2))
        conv_a4 = F.elu(self.conv2d4(conv_a3))
        conv_a4_reshape = conv_a4.view(-1, 1024)
        conv_a4_y = torch.cat((conv_a4_reshape, y), -1)
        linear_a1 = F.elu(self.linear1(conv_a4_y))
        linear_a2 = torch.sigmoid(self.linear2(linear_a1))
        return linear_a2

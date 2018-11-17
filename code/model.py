# TODO: perform any necessary preprocessing on x (like scale pixel values to [-1, 1], etc)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        # TODO: Add padding for nn.ConvTranspose2d (as if we were using padding "same")
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 1024) # Restriction: nn.Linear(b, a) -> nn.Linear(c, b)
        self.deconv2d1 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2) # Input is (8, 8, 64), output is (16, 16, 32)
        self.deconv2d2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2) # Input is (16, 16, 32), output is (32, 32, 32)
        self.deconv2d3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2) # Input is (32, 32, 32), output is (64, 64, 32)
        self.deconv2d4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2) # Input is (64, 64, 32), output is (128, 128, 3)

    # x is (m, 1024), y is (m, n_classes)
    def forward(self, z, y):
        zy = torch.cat((z, y), -1)
        linear_a1 = F.elu(self.linear1(zy))
        linear_a2 = F.elu(self.linear2(linear_a1))
        linear_a2_reshape = linear_a2.view(-1, 8, 8, 16)
        deconv_a1 = F.elu(self.deconv2d1(linear_a2_reshape))
        deconv_a2 = F.elu(self.deconv2d2(deconv_a1))
        deconv_a3 = F.elu(self.deconv2d3(deconv_a2))
        deconv_a4 = F.sigmoid(self.deconv2d4(deconv_a3))
        return deconv_a4

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        # Layers
        self.conv2d1 = nn.Conv2d(3, 32, kernel_size=5, stride=2) # Input is (128, 128, 3), output is (64, 64, 32)
        self.conv2d2 = nn.Conv2d(32, 32, kernel_size=5, stride=2) # Input is (64, 64, 32), output is (32, 32, 32)
        self.conv2d3 = nn.Conv2d(32, 32, kernel_size=5, stride=2) # Input is (32, 32, 32), output is (16, 16, 32)
        self.conv2d4 = nn.Conv2d(32, 64, kernel_size=5, stride=2) # Input is (16, 16, 32), output is (8, 8, 64)
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
        linear_a2 = F.sigmoid(self.linear2(linear_a1))
        return linear_a2

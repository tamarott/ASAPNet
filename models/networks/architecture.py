"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE
from models.networks.normalization import slimSPADE
from models.networks.condconv import DepthConv
import functools
import numpy as np

class ASAPNetsResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(True), kernel_size=3):
        super().__init__()

        self.conv_block = nn.Sequential(
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1)),
            activation
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out

class ASAPNetsBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(True), kernel_size=3, reflection_pad=False, replicate_pad=False):
        super().__init__()
        padw = 1
        if reflection_pad:
            self.conv_block = nn.Sequential(nn.ReflectionPad2d(padw),
                                            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=0)),
                                            activation
                                            )
        elif replicate_pad:
            self.conv_block = nn.Sequential(nn.ReplicationPad2d(padw),
                                            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=0)),
                                            activation
                                            )
        else:
            self.conv_block = nn.Sequential(norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padw)),
                                            activation
                                            )

    def forward(self, x):
        out = self.conv_block(x)
        return out


class ASAPNetsGradBlock(nn.Module):
    def __init__(self, dim_in, dim_out, norm_layer, activation=nn.ReLU(True), kernel_size=3, reflection_pad=False):
        super().__init__()
        padw = 1
        if reflection_pad:
            self.conv_block = nn.Sequential(nn.ReflectionPad2d(padw),
                                            norm_layer(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=0)),
                                            activation
                                            )
        else:
            self.conv_block = nn.Sequential(norm_layer(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padw)),
                                            activation
                                            )

    def forward(self, x):
        out = self.conv_block(x)
        return out



# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class MySeparableBilinearDownsample(torch.nn.Module):
    def __init__(self, stride, channels, use_gpu):
        super().__init__()
        self.stride = stride
        self.channels = channels

        # create tent kernel
        kernel = np.arange(1,2*stride+1,2) # ramp up
        kernel = np.concatenate((kernel,kernel[::-1])) # reflect it and concatenate
        if use_gpu:
            kernel = torch.Tensor(kernel/np.sum(kernel)).to(device='cuda') # normalize
        else:
            kernel = torch.Tensor(kernel / np.sum(kernel))
        self.register_buffer('kernel_horz', kernel[None,None,None,:].repeat((self.channels,1,1,1)))
        self.register_buffer('kernel_vert', kernel[None,None,:,None].repeat((self.channels,1,1,1)))

        self.refl = nn.ReflectionPad2d(int(stride/2))#nn.ReflectionPad2d(int(stride/2))

    def forward(self, input):
        return F.conv2d(F.conv2d(self.refl(input), self.kernel_horz, stride=(1,self.stride), groups=self.channels),
                    self.kernel_vert, stride=(self.stride,1), groups=self.channels)

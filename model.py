#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:50:36 2019

@author: hazem
"""

import torch.nn as nn

class GNetInitialBlock(nn.Module):
    def __init__(self, n_out_c):
        super(GNetInitialBlock, self).__init__()
        
        # c7s1-k: 7x7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1
        # k = n_out_c
        layers = [   nn.ReflectionPad2d(3),
                     nn.Conv2d(3, n_out_c, 7),
                     nn.InstanceNorm2d(n_out_c),
                     nn.ReLU(inplace=True) ]
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)
    
class GNetFinalBlock(nn.Module):
    def __init__(self, n_in_c):
        super(GNetFinalBlock, self).__init__()
        
        # c7s1-3: 7x7 Convolution-Tanh layer with 3 filters and stride 1
        layers = [   nn.ReflectionPad2d(3),
                     nn.Conv2d(n_in_c, 3, 7),
                     nn.Tanh() ]
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)
    
class DownBlock(nn.Module):
    def __init__(self, n_in_c, n_out_c):
        super(DownBlock, self).__init__()
        
        # dk: 3x3 Convolution-InstanceNorm-ReLU layers with k filters and stride 2
        # k = n_out_c
        layers = [   nn.Conv2d(n_in_c, n_out_c, 3, stride=2, padding=1),
                     nn.InstanceNorm2d(n_out_c),
                     nn.ReLU(inplace=True) ]
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)
    
class UpBlock(nn.Module):
    def __init__(self, n_in_c, n_out_c):
        super(UpBlock, self).__init__()
        
        # uk: 3x3 Deconvolution-InstanceNorm-ReLU layers with k filters and stride 2
        # k = n_out_c
        layers = [   nn.ConvTranspose2d(n_in_c, n_out_c, 3, stride=2, padding=1, output_padding=1),
                     nn.InstanceNorm2d(n_out_c),
                     nn.ReLU(inplace=True) ]
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)
    
class ResBlock(nn.Module):
    def __init__(self, n_c):
        super(ResBlock, self).__init__()
        
        # Rk: Two 3 × 3 convolutional layers with k filters
        # k = n_c
        layers = [  nn.ReflectionPad2d(1),
                    nn.Conv2d(n_c, n_c, 3),
                    nn.InstanceNorm2d(n_c),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(n_c, n_c, 3),
                    nn.InstanceNorm2d(n_c) ]
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return x + self.block(x)

class DNetConvBlock(nn.Module):
    def __init__(self, n_in_c, n_out_c, s=2, norm=True):
        super(DNetConvBlock, self).__init__()
        
        # Ck: 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2
        # k = n_out_c
        layers = [nn.Conv2d(n_in_c, n_out_c, 4, stride=s, padding=1)]
        
        if norm:
            layers += [nn.InstanceNorm2d(n_out_c)]
            
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)
    
class GNet(nn.Module):
    def __init__(self, n_init=64, depth=2, width=9):
        super(GNet, self).__init__()
        
        blocks = []
        
        # Initial Block: 7x7 Convolution-InstanceNorm-ReLU layer with stride 1
        blocks += [GNetInitialBlock(n_init)]
        
        n_c = n_init
        
        # Downsampling: 3x3 Convolution-InstanceNorm-ReLU layers with stride 2
        for _ in range(depth):
            blocks += [DownBlock(n_c, n_c*2)]
            n_c *= 2
        
        # Residual blocks: Each has two 3 × 3 convolutional layers
        for _ in range(width):
            blocks += [ResBlock(n_c)]
        
        # Upsampling: 3 × 3 Deconvolution-InstanceNorm-ReLU layers
        for _ in range(depth):
            blocks += [UpBlock(n_c, n_c//2)]
            n_c //= 2
        
        # Final Block: 7x7 Convolution-Tanh layer with stride 1
        blocks += [GNetFinalBlock(n_c)]
        
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)
    
class DNet(nn.Module):
    def __init__(self, n_init=64, depth=4):
        super(DNet, self).__init__()
        
        blocks = []
        
        # A few 4 × 4 Convolution-InstanceNorm-LeakyReLU blocks with stride 2
        # First convolution block does not use InstanceNorm
        blocks += [DNetConvBlock(3, n_init, norm=False)]
        
        n_c = n_init
        
        # The middle convolution blocks
        for _ in range(depth - 2):
            blocks += [DNetConvBlock(n_c, n_c*2)]
            n_c *= 2
            
        # Last convolution block uses stride=1
        blocks += [DNetConvBlock(n_c, n_c*2, s=1)]
        n_c *= 2
        
        
        # One last convolution to produce a 1-dimensional output
        blocks += [nn.Conv2d(n_c, 1, 4, padding=1)]
        
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

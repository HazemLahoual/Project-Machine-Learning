#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:57:00 2019

@author: hazem
"""
import torch
import random

def weights_init_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
        
class LR_Decay_Multiplier():
    def __init__(self, n_epochs, decay_epoch):
        self.n_epochs = n_epochs
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        if (epoch < self.decay_epoch):
            decay = 0.0
        else:
            decay = (epoch+1 - self.decay_epoch)/(self.n_epochs - self.decay_epoch)
        return 1.0 - decay

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)
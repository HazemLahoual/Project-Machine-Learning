#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Nov  9 23:17:26 2019

@author: hazem
"""
import argparse
import itertools
import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from model import GNet
from model import DNet

from utils import LR_Decay_Multiplier
from utils import ReplayBuffer
from utils import weights_init_normal

from dataset import load_data


parser = argparse.ArgumentParser(prog='PROG',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset-related
parser.add_argument('--dataroot', type=str, default='/home/space/datasets/Unpaired-I2I-translation', help='root directory of dataset')
parser.add_argument('--dataset', type=str, default='landscape2monet', help='dataset')
parser.add_argument('--folder_A', type=str, default='A', help='name of directory for dataset A in dataroot')
parser.add_argument('--folder_B', type=str, default='B', help='name of directory for dataset B in dataroot')
parser.add_argument('--n_train_A', type=float, default=0.9, help='number of train samples from dataset A. If n >= 1.0, then the specified number will be converted to int and used as exact number of samples to be used')
parser.add_argument('--n_test_A', type=float, default=0.1, help='number of test samples from dataset A. If n >= 1.0, then the specified number will be converted to int and used as exact number of samples to be used')
parser.add_argument('--n_train_B', type=float, default=0.9, help='number of train samples from dataset B. If n >= 1.0, then the specified number will be converted to int and used as exact number of samples to be used')
parser.add_argument('--n_test_B', type=float, default=0.1, help='number of test samples from dataset B. If n >= 1.0, then tthe specified number will be converted to int and used as exact number of samples to be used')
parser.add_argument('--model', type=str, default='model', help='name of model trained')
parser.add_argument('--reuse_model', type=str, default=None, help='name of the model, which we want to use its saved splitted datasets')
parser.add_argument('--resize_size', type=int, default=286, help='resize of training images to <resize_size x resize_size>')
parser.add_argument('--crop_size', type=int, default=256, help='scrop the image to <crop_size x crop_size>')
parser.add_argument('--BW', action='store_false', help='Black&White/Grayscale images removal (if --BW is used then there will be no grayscale images removal')
parser.add_argument('--resize', action='store_false', help='resize the images to a given value (if --resize is used then there will be no resize transformation)')
parser.add_argument('--crop', action='store_false', help='random crop the images to a given value (if --crop is used then there will be no crop transformation)')
parser.add_argument('--flip', action='store_false', help='random horizontal flip (if --flip is used then there will be no flip transformation)')
parser.add_argument('--normalize', action='store_false', help='normalize the images (if --normalize is used then there will be no normalization transformation)')

# Model-architecture-related
parser.add_argument('--G_init_filter', type=int, default=64, help='number of filters in first layer of generator')
parser.add_argument('--G_depth', type=int, default=2, help='number of downsampling/upsampling blocks in generator')
parser.add_argument('--G_width', type=int, default=9, help='number of residual blocks in generator')
parser.add_argument('--D_init_filter', type=int, default=64, help='number of filters in first layer of discriminator')
parser.add_argument('--D_depth', type=int, default=4, help='number of convolutional blocks for discriminator')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')

# Training-scheme-related
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs for training')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which learning rate decreases to 0 linearly')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--lambda_', type=float, default=10.0, help='hyperparameter controlling the relative importance of the losses')
parser.add_argument('--identity_loss', action='store_false', help='add identity loss in cost function')
parser.add_argument('--buffer_size', type=int, default=50, help='size of buffer used to train discriminator')

parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

parser.add_argument('--frequency',type=int, default=5, help='frequency of saving model\'s parameters')

opt = parser.parse_args()

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
Gnet_AB = GNet(opt.G_init_filter, opt.G_depth, opt.G_width)
Gnet_BA = GNet(opt.G_init_filter, opt.G_depth, opt.G_width)
Dnet_A = DNet(opt.D_init_filter, opt.D_depth)
Dnet_B = DNet(opt.D_init_filter, opt.D_depth)

if opt.cuda:
    Gnet_AB.cuda()
    Gnet_BA.cuda()
    Dnet_A.cuda()
    Dnet_B.cuda()

# Weight Initialization from a Gaussian distribution N(0, 0:02)
Gnet_AB.apply(weights_init_normal)
Gnet_BA.apply(weights_init_normal)
Dnet_A.apply(weights_init_normal)
Dnet_B.apply(weights_init_normal)

# Lossess
L_GAN = nn.MSELoss()
L_cyc = nn.L1Loss()
L_identity = nn.L1Loss()

if opt.cuda:
    L_GAN.cuda()
    L_cyc.cuda()
    L_identity.cuda()
    
if opt.identity_loss:
    lambda_identity = opt.lambda_/2
else:
    lambda_identity = 0

# Optimizers
optim_G = Adam(itertools.chain(Gnet_AB.parameters(), Gnet_BA.parameters()),
               lr=opt.lr, betas=(0.5, 0.999))
optim_D = Adam(itertools.chain(Dnet_A.parameters(), Dnet_B.parameters()),
               lr=opt.lr, betas=(0.5, 0.999))

# Learning rate schedulers
lr_scheduler_G = LambdaLR(optim_G, lr_lambda=LR_Decay_Multiplier(opt.n_epochs, opt.decay_epoch).step)
lr_scheduler_D = LambdaLR(optim_D, lr_lambda=LR_Decay_Multiplier(opt.n_epochs, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

if opt.resize:
    if opt.crop:
        input_A = Tensor(opt.batchSize, opt.input_nc, opt.crop_size, opt.crop_size)
        input_B = Tensor(opt.batchSize, opt.output_nc, opt.crop_size, opt.crop_size)
    else:
        input_A = Tensor(opt.batchSize, opt.input_nc, opt.resize_size, opt.resize_size)
        input_B = Tensor(opt.batchSize, opt.output_nc, opt.resize_size, opt.resize_size)
elif opt.crop:
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.crop_size, opt.crop_size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.crop_size, opt.crop_size)       
else:
    assert opt.resize or opt.crop ,"All the inputs should have common size. Please use resize and/or crop transformations"


real_label = Tensor([1.0])
fake_label = Tensor([0.0])

Buffer_A = ReplayBuffer()
Buffer_B = ReplayBuffer()

transform_list = []
if opt.resize:
     transform_list.append('resize')
if opt.crop:
     transform_list.append('crop')
if opt.flip:
    transform_list.append('flip')
if opt.normalize:
    transform_list.append('normalize')

if opt.n_train_A >= 1.0:
    opt.n_train_A = int(opt.n_train_A)
if opt.n_test_A >= 1.0:
    opt.n_test_A = int(opt.n_test_A)
if opt.n_train_B >= 1.0:
    opt.n_train_B = int(opt.n_train_B)
if opt.n_test_B >= 1.0:
    opt.n_test_B = int(opt.n_test_B)
    
dataloader = load_data(dataset= opt.dataset, name_model= opt.model, root = opt.dataroot, mode = 'load_dataset', reuse = opt.reuse_model, opt = opt, n_train=[opt.n_train_A,opt.n_train_B], n_test=[opt.n_test_A,opt.n_test_B], transformation=transform_list, load_size=opt.resize_size, crop_size=opt.crop_size)

path_output = os.path.join("output", opt.model)
if not os.path.exists(path_output):
    os.makedirs(path_output, exist_ok=True)

print('\n|---> the model\'s parameters and  errors will be saved in <current folder>/output/'+opt.model+'\n')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

print('=========================================')
print('----------------- Training --------------')
print('=========================================')

loss_i_A = []
loss_i_B = []
loss_gan_ab = []
loss_gan_ba = []
loss_c_aba = []
loss_c_bab = []
loss_g = []
loss_d_a = []
loss_d_b = []
    
time_epoch = []
total_time = time.time()

for epoch in range(1, opt.n_epochs+1):
    epoch_start = time.time()
    for i, batch in enumerate(dataloader): 
        if i == len(dataloader)-1:
            print("|--- epoch: {0}/{1} | lr: {2:0.6f} | batch: {3}/{4}".format(epoch, opt.n_epochs,get_lr(optim_G), i+1, len(dataloader))+" --- done!")
        else:
            print("|--- epoch: {0}/{1} | lr: {2:0.6f} | batch: {3}/{4}".format(epoch, opt.n_epochs, get_lr(optim_G), i+1, len(dataloader)), end='\r')
        #define inputs
        real_A = input_A.copy_(batch['A'])
        real_B = input_B.copy_(batch['B'])
        
        optim_G.zero_grad()
        
        #identity loss
        BB = Gnet_AB(real_B)
        L_identity_BB = L_identity(real_B, BB) * lambda_identity
        
        AA = Gnet_BA(real_A)
        L_identity_AA = L_identity(real_A, AA) * lambda_identity

        #GAN loss
        AB = Gnet_AB(real_A)
        AB_realB = Dnet_B(AB)
        L_GAN_AB = L_GAN(AB_realB, real_label.expand_as(AB_realB))
        
        BA = Gnet_AB(real_B)
        BA_realA = Dnet_A(BA)
        L_GAN_BA = L_GAN(BA_realA, real_label.expand_as(AB_realB))

        
        #Cycle loss
        ABA = Gnet_BA(AB)
        L_cyc_ABA = L_cyc(ABA, real_A) * opt.lambda_
        
        BAB = Gnet_AB(BA)
        L_cyc_BAB = L_cyc(BAB, real_B) * opt.lambda_

        #Total loss
        L_G = L_GAN_AB + L_GAN_BA + L_cyc_ABA + L_cyc_BAB + L_identity_AA + L_identity_BB
        
        L_G.backward()
        
        optim_G.step()

        

        #discriminator A
        A_realA = Dnet_A(real_A)
        loss_D_real = L_GAN(A_realA, real_label.expand_as(A_realA))
        
        BA_from_Buffer = Buffer_A.push_and_pop(BA)
        pred_fake = Dnet_A(BA_from_Buffer.detach())
        loss_D_fake = L_GAN(pred_fake, fake_label.expand_as(pred_fake))

        loss_D_A = (loss_D_real + loss_D_fake)*0.5

        #discriminator B
        B_realB = Dnet_B(real_B)
        loss_D_real = L_GAN(B_realB, real_label.expand_as(B_realB))
        
        AB_from_Buffer = Buffer_B.push_and_pop(AB)
        pred_fake = Dnet_B(AB_from_Buffer.detach())
        loss_D_fake = L_GAN(pred_fake, fake_label.expand_as(pred_fake))

        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        
        optim_D.zero_grad()
        loss_D_A.backward()
        loss_D_B.backward()
        optim_D.step()

    #update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    
    loss_i_B.append(L_identity_BB) 
    loss_i_A.append(L_identity_AA) 
    loss_gan_ab.append(L_GAN_AB)
    loss_gan_ba.append(L_GAN_BA)
    loss_c_aba.append(L_cyc_ABA)
    loss_c_bab.append(L_cyc_BAB)
    loss_g.append(L_G)
    loss_d_a.append(loss_D_A)
    loss_d_b.append(loss_D_B)
    
    epoch_end = time.time()
    time_epoch.append(epoch_end-epoch_start)
    print("epoch: {0}/{1} | G_loss = {2:.2f} | D_loss = {3:.2f}".format(epoch, opt.n_epochs,L_G, loss_D_A+loss_D_B))
    print("Time required to compute: {0:.2f}s | total time: {1:.2f}s".format(epoch_end-epoch_start, time.time()-total_time))
    #save model's parameters
    if ((epoch%opt.frequency) == 0):
        torch.save(Gnet_AB.state_dict(), os.path.join(path_output,'Gnet_AB-')+str(epoch)+'.pth')
        torch.save(Gnet_BA.state_dict(), os.path.join(path_output,'Gnet_BA-')+str(epoch)+'.pth')
        torch.save(Dnet_A.state_dict(), os.path.join(path_output,'Dnet_A-')+str(epoch)+'.pth')
        torch.save(Dnet_B.state_dict(), os.path.join(path_output,'Dnet_B-')+str(epoch)+'.pth')
                   
np.save(os.path.join(path_output,"loss_i_A"), np.array(loss_i_A))
np.save(os.path.join(path_output,"loss_i_B"), np.array(loss_i_B))
np.save(os.path.join(path_output,"loss_gan_ab"), np.array(loss_gan_ab))
np.save(os.path.join(path_output,"loss_gan_ba"), np.array(loss_gan_ba))
np.save(os.path.join(path_output,"loss_c_aba"), np.array(loss_c_aba))
np.save(os.path.join(path_output,"loss_c_bab"), np.array(loss_c_bab))
np.save(os.path.join(path_output,"loss_g"), np.array(loss_g))
np.save(os.path.join(path_output,"loss_d_a"), np.array(loss_d_a))
np.save(os.path.join(path_output,"loss_d_b"), np.array(loss_d_b))
np.save(os.path.join(path_output,"time_epoch"), np.array(time_epoch))

print("\n---> Training time: {}".format(time.time()-total_time))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 02:23:35 2019

@author: hazem
"""

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
import csv
import os
import os.path
import numpy as np
from PIL import Image
import random
from sklearn.model_selection import train_test_split
    
def load_data(dataset, name_model, root='.', mode='load_generator', reuse=None, opt = None , 
              n_train=[0.9,0.9], n_test=[0.1,0.1], transformation=['resize', 'crop', 'flip', 'normalize'], 
              load_size=286, crop_size=256):
    
    """
    Return two generators or dataloader depends on the mode chosen.
    
    Parameters:
------- dataset (str)[Required]         -- name of the dataset to load (e.g. 'horse2zebra', 'landscape2vangogh'...). The dataset should have 2 subfolders: 
        - <dataset>/A : that contains the images of the first class of images
        - <dataset>/B: that contains the images of the second class of images 
        [e.g. in 'horse2zebra' dataset, subfolder 'A' contains images of horses, subfolder 'B' contains images of zebras]
                                 
------- name_model (str)[required]      -- a giving name to the model (as it's needed to save model's parameters later, and to make the reuse of the model possible)
                                 
------- root (str)[optional]            -- root of the datasets folder (which can contains one or many datasets) (e.g. '/space/datasets/Unpaired_I2I')
                                 
------- mode (str)[optional]            -- two possible modes: 
        'load_generator' | 'load_dataset': 
        -'load_generator': will return two generators, one that will generate image from train set, the other will generate images from test set. 
        -'load_dataset': the function will return a dataset of class DataLoader that can be used for the training.
                                   
------- reuse (str)[optional]           -- name of the model to reuse (e.g. reuse='model1' to reuse the splitting of dataset of model1)
        
------- opt (parser)[optional]          -- a parser that contains all options passed through command line.
        
------- n_train (list)[optional]        -- a list of the form [number of train samples for 'A', number of train samples for 'B']. n_train could be int or fload numbers (in case of float numbers, n_train sould be 0 < n_train[i] < 1. The ratio of n_train[i] from all samples will be used to split the dataset).
        (e.g. n_train=[0.9, 1000]: for class A, 90% of samples will be used as train set, for class B, 1000 random samples will be chosen as train set. The n_test should have the form as n_train. That means in our case n_test = [0.1, <init>])
                                 
------- n_test (list)[optional]         -- a list of the form [number of test samples for 'A', number of test samples for 'B']. n_test could be int or fload numbers (in case of float numbers, n_test sould be 0 < n_test[i] < 1 for i in {0,1}. The ratio of n_test from all samples will be used to split the dataset)
                                 
------- transformation (list[optional]) -- list of transformations to apply. Possbile transformations are 'resize' | 'crop' | 'flip' | 'normalize'. 
        -'resize':resize the image to <load_size>.
        -'crop': random crop to <crop_size>. 
        -'flip': random horizontal flip. 
        -'normalize': normalize the image to [-1,1].
                                 
------- load_size (int)[optional]       -- resize the images to load_size
        
------- crop_size (int)[optional]       -- random crop of the images to crop_size 

---> Returns in case of:
------- mode: "load_generator": 
        generaor_im_train (class Generator)       -- a generator of train images. When called will return a list of the form [image from train set of "A", image from train set of "B]
        generator_im_test (class Generator)       -- a generator of test images. When called will return a list of the form [image from test set of "A", image from test set of "B]

------- mode: "load_dataset":
        dataset (class DataLoader)       -- a dataset to use for the training phase
    """


    #set /path/to/<dataset>
    path = os.path.join(root,dataset)
    #set /path/to/<dataset>/saved_models: the folder where to save the models used
    path_saved_models = os.path.join(path,'saved_models')
    print('=========================================')
    #check if the name of the nodel is used before
    avail = check_name_availability(path_saved_models, name_model)
    if avail == False:
        print("|---> name of the model '"+name_model+"' is already used. This model will replace the old one")
    else:
        print("|---> the given name for the model is not used before. A new folder "+ os.path.join(path_saved_models, name_model)+" is created")
    print('=========================================\n')
    #print the basic options
    message_base_opt(root, dataset, path, path_saved_models, name_model, reuse, mode)

    print('=========================================')
    print('-------------- Preprocessing ------------')
    print('=========================================')
    #check first if an old model will be reused
    if reuse == None:
        #check if is called to return generators or to train a network and if the the operation of removing grayscale images is True
        if opt == None or opt.BW == True:
            print('|---> removing black&white images')
            #remove grayscale images from class A
            list_without_BW_im_a, n_bw_im_a = remove_BW_images(path, "A")
            print('      |---> {} images removed from set \"A\"'.format(n_bw_im_a))
            #remove grauscale images from class B
            list_without_BW_im_b, n_bw_im_b = remove_BW_images(path, "B")
            print('      |---> {} images removed from set \"B\"'.format(n_bw_im_b))
            #prepare list of final list filenames of images
            list_im_A = list_without_BW_im_a
            list_im_B = list_without_BW_im_b
        else:
            #in case of remove grayscale images is False, just get the filename of all images
            print('|---> removing black&white images: BW = False')
            list_im_A = get_all_images_path(path, "A")
            list_im_B = get_all_images_path(path, "B")
        
        path_dataset_A = list_im_A
        path_dataset_B = list_im_B
        #split the dataset into train and test sets
        train_A, test_A = split_dataset(path_dataset_A, n_test[0], n_train[0])
        train_B, test_B = split_dataset(path_dataset_B, n_test[1], n_train[1])
        #print informations about datasets
        message_split(train_A, test_A, train_B, test_B, n_train, n_test)
        #save the list of filnenames of train and tset splits into /path/to/<dataset>/saved_models
        save_datasets(path_saved_models, name_model, train_A, test_A, train_B, test_B)
    else:
        print("|---> Reusing model {} datasets".format(reuse))
        list_im_A = get_all_images_path(path, "A")
        list_im_B = get_all_images_path(path, "B")
        list_saved_models = get_names_list(path_saved_models)
        if reuse not in list_saved_models:
            print("|---> error: name of model {} not found. No model will be reused.".format(reuse))
            
            if opt == None or opt.BW == True:
                print('|---> removing black&white images')
                list_without_BW_im_a, n_bw_im_a = remove_BW_images(path, "A")
                print('      |---> {} images removed from set \"A\"'.format(n_bw_im_a))
                list_without_BW_im_b, n_bw_im_b = remove_BW_images(path, "B")
                print('      |---> {} images removed from set \"B\"'.format(n_bw_im_b))      
                list_im_A = list_without_BW_im_a
                list_im_B = list_without_BW_im_b
            else:
                print('|---> removing black&white images: BW = False')
                list_im_A = get_all_images_path(path, "A")
                list_im_B = get_all_images_path(path, "B")
            
            path_dataset_A = list_im_A
            path_dataset_B = list_im_B
            
            train_A, test_A = split_dataset(path_dataset_A, n_test[0], n_train[0])
            train_B, test_B = split_dataset(path_dataset_B, n_test[1], n_train[1])
            
            message_split(train_A, test_A, train_B, test_B, n_train, n_test)
            
            save_datasets(path_saved_models, name_model, train_A, test_A, train_B, test_B)
        else:
            print('|---> loading the saved datasets of {}'.format(reuse))
            train_A, test_A, train_B, test_B = load_datasets(path_saved_models, reuse)
            message_split(train_A, test_A, train_B, test_B, n_train, n_test)
            save_datasets(path_saved_models, name_model, train_A, test_A, train_B, test_B)
            
    transformation_to_apply = get_transformations(transformation)
    message_transformations(transformation, load_size, crop_size)
    print("========================================\n")
    
    if mode == 'load_generator':
        generator_im_train = Generator(path, "train",train_A, test_A, train_B, test_B, transformation_to_apply, transformation)
        generator_im_test = Generator(path, "test", train_A, test_A, train_B, test_B, transformation_to_apply, transformation)
        print("---> Generators created")
        return generator_im_train, generator_im_test
    
    elif mode == 'load_dataset':
        dataloader = DataLoader(ImageDataset(root, dataset, train_A, train_B, transforms_=transformation_to_apply),
                                batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
        print("---> dataloader created\n")
        message_model_train_opt(opt)
        return dataloader
    else:
        print("mode not recognized")

    
class ImageDataset(Dataset):
    def __init__(self, root, dataset, train_A, train_B, transforms_=None) :
        self.transform = transforms_
        self.files_A = add_path_to_lists(root, train_A, os.path.join(dataset, 'A'))
        self.files_B = add_path_to_lists(root, train_B, os.path.join(dataset, 'B'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

def get_names_list(path):
    return [f.name for f in os.scandir(path) if f.is_dir()]

def check_name_availability(path_saved_models, name_model):
    if not os.path.exists(path_saved_models):
        os.makedirs(os.path.join(path_saved_models, name_model))
        return True
    else:
        list_saved_names = get_names_list(path_saved_models)
        print("|---> list of saved models: ", list_saved_names)
        if name_model in list_saved_names:
            return False
        else:
            os.makedirs(os.path.join(path_saved_models, name_model))
            return True
        
def check_models_dir(path, name):
    if not os.path.exists(os.path.join(path,name)):
        os.makedirs(os.path.join(path,name))

def get_all_images_path(dir, dataset):
    path_images_list = []
    for root, dirs, files in sorted(os.walk(dir, topdown=True)):
        for name in dirs:
            if name == dataset:
                for filename in sorted(os.listdir(os.path.join(root,name))):
                        path_images_list.append(filename)                                
    return np.asarray(path_images_list)
              
def split_dataset(dataset, n_train, n_test):
    set_train, set_test= train_test_split(dataset, test_size=n_train,train_size=n_test)
    return set_train, set_test

      
def check_RGB_images(image_path):
    im = Image.open(image_path)
    im2arr = np.array(im)
    if len(im2arr.shape) == 3:
        return True
    else:
        return False

def add_path_to_lists_BW(path, set_, list_to_add):
    to_return = []
    for i in list_to_add:
        to_return.append(os.path.join(path,set_, i))
    return to_return

def remove_BW_images(path, set_to_remove_from):
    to_return = []
    number_BW_im = 0
    list_images_in_set = get_all_images_path(path, set_to_remove_from)
    complete_list= add_path_to_lists_BW(path, set_to_remove_from, list_images_in_set)
    for i,j in enumerate(complete_list):
        if i == len(complete_list)-1:
            print("|---> "+ set_to_remove_from+": checking image {0}/{1}".format(i+1, len(complete_list))+" --- done!")
        else:
            print("|---> "+set_to_remove_from+": checking image {0}/{1}".format(i+1, len(complete_list)), end='\r')
        if check_RGB_images(j):
            to_return.append(list_images_in_set[i])
        else:
            number_BW_im += 1
    return to_return, number_BW_im
  
def get_transformations(transformation = ['resize', 'crop', 'flip', 'normalize'], load_size = 286, crop_size = 256, method=Image.BICUBIC):
    transform_list = []
    if 'resize' in transformation:
        resize_to = [load_size, load_size]
        transform_list.append(transforms.Resize(resize_to, method))
    if 'crop' in transformation:
        transform_list.append(transforms.RandomCrop(crop_size))
    if 'flip' in transformation:
        transform_list.append(transforms.RandomHorizontalFlip())
   
    if 'normalize' in transformation:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def add_path_to_lists(path, list_to_add, mode):
    to_return = []
    for i in list_to_add:
        to_return.append(os.path.join(path,mode,i))
    return to_return

def save_a_list(path, list_to_save, name_list):
    with open(os.path.join(path, name_list), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(list_to_save)
        
def save_datasets(path, name, train_A, test_A, train_B, test_B):
    check_models_dir(path, name)
    save_datasets_to_path  = os.path.join(path, name)
    save_a_list(save_datasets_to_path, train_A, 'train_A')
    save_a_list(save_datasets_to_path, test_A, 'test_A')
    save_a_list(save_datasets_to_path, train_B, 'train_B')
    save_a_list(save_datasets_to_path, test_B, 'test_B')

def load_list_from_saved(path, mode):
    to_return = []
    read_list = []
    with open(os.path.join(path, mode), newline='') as csvfile:
        l = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in l:
            read_list.append(row)
    
    for i in (read_list[0]):
        to_return.append(i)
    return to_return

def load_datasets(path_saved_models, reuse):
    path_lists_to_load = os.path.join(path_saved_models, reuse)
    train_A = load_list_from_saved(path_lists_to_load, 'train_A')
    test_A = load_list_from_saved(path_lists_to_load, 'test_A')
    train_B = load_list_from_saved(path_lists_to_load, 'train_B')
    test_B = load_list_from_saved(path_lists_to_load, 'test_B')
    return train_A, test_A, train_B, test_B

class Generator:
    def __init__(self, newpath, dataset_to_generate_from,train_A, test_A, train_B, test_B, transformation, list_transformations):
        self.newpath = newpath
        self.dataset_to_generate_from = dataset_to_generate_from
        self.transform = transformation
        self.list_transformations = list_transformations

        if self.dataset_to_generate_from == 'train':
            self.list_a = add_path_to_lists(newpath, train_A, "A")
            self.list_b = add_path_to_lists(newpath, train_B, "B")
            self.buffer_a = []
            self.buffer_b = []
            
        if self.dataset_to_generate_from == 'test':
            self.list_a = add_path_to_lists(newpath, test_A, "A")
            self.list_b = add_path_to_lists(newpath, test_B, "B")
            self.buffer_a = add_path_to_lists(newpath, test_A, "A")
            self.buffer_b = add_path_to_lists(newpath, test_B, "B")
            
    def __call__(self):
        if self.dataset_to_generate_from == "test":       
            index_a = np.random.choice(len(self.buffer_a))
            index_b = np.random.choice(len(self.buffer_b))
            if 'normalize' in self.list_transformations:
                to_return = [np.array(self.transform(Image.open(self.buffer_a[index_a]))).transpose(2,1,0).transpose(1,0,2), np.array(self.transform(Image.open(self.buffer_b[index_b]))).transpose(2,1,0).transpose(1,0,2)]
            else:
                to_return = [np.array(self.transform(Image.open(self.buffer_a[index_a]))), np.array(self.transform(Image.open(self.buffer_b[index_b])))]
            self.buffer_a.remove(self.buffer_a[index_a])
            self.buffer_b.remove(self.buffer_b[index_b])
            if self.buffer_a == [] and self.buffer_b != []:
                print("All images from TestA were returned. BufferA will reset")
                self.buffer_a = list.copy(self.list_a)
            elif self.buffer_a != [] and self.buffer_b == []:
                print("All images from TestB were returned. BufferB will reset")
                self.buffer_b = list.copy(self.list_b)
            elif self.buffer_a == [] and self.buffer_b == []:
                print("All images from testA and testA were returned. BufferA and BufferB will reset")
                self.buffer_a=list.copy(self.list_a)
                self.buffer_b = list.copy(self.list_b)
            
            return to_return

        if self.dataset_to_generate_from =="train":
            index_a = np.random.choice(len(self.list_a))
            index_b = np.random.choice(len(self.list_b))
            self.buffer_a.append(self.list_a[index_a])
            self.buffer_b.append(self.list_b[index_b])
            if 'normalize' in self.list_transformations:
                return [np.array(self.transform(Image.open(self.list_a[index_a]))).transpose(2,1,0).transpose(1,0,2), np.array(self.transform(Image.open(self.list_b[index_b]))).transpose(2,1,0).transpose(1,0,2)]
            else:
                return [np.array(self.transform(Image.open(self.list_a[index_a]))), np.array(self.transform(Image.open(self.list_b[index_b])))]

def message_transformations(transformation, load_size, crop_size):
    message = ""
    if transformation==None or transformation==[]:
        message += '|--- Transformation = None\n'
    else:
        t_s = ''
        for s in transformation:
            t_s += s + ' | '
        message += '|---> Transformations = {}\n '.format(t_s)
        if 'resize' in transformation:
            message += '     |---> resize to = {0}\n'.format(load_size)
        if 'crop' in transformation:
            message += '     |---> crop size = {}\n'.format(crop_size)
        if 'flip' in transformation:
            message += '     |---> Random horizontal flip\n'
        if 'normalize' in transformation:
            message += '     |---> Normalize images to [-1,1] \n'
    message += '[The transformations are not applied to the original dataset, but it will only be applied when the generators/dataloader are called]'
    print(message)
      
def message_base_opt(root, dataset, path, path_saved_models, name_model, reuse, mode):
    print('=========================================')
    print('-------------- Base Options -------------')
    print('=========================================')
    print('|--- root: {}'.format(root+'/'))
    print('|--- dataset: {0} [path: {1}]'.format(dataset, os.path.join(root,dataset)))
    print('     |--- divided into --- A: contains {} images'.format(len(get_all_images_path(path, "A"))))
    print('                       |--- B: contains {} images'.format(len(get_all_images_path(path, "B"))))
    print("|--- model name: {0}".format(name_model))
    print("|--- model will be saved [only splitted datasets, not transformations] in: {}.".format(path_saved_models))
    print('|--- reusing another model: {}'.format(reuse))
    print('|--- mode: {}'.format(mode))
    print('=========================================\n')
    
def message_split(train_A, test_A, train_B, test_B, n_train, n_test):
    if type(n_train[0])==float:
        print('|---> Splitting dataset into --- train_A: {} [ratio: {}]'.format(len(train_A), n_train[0]))
    else:
        print('|---> Splitting dataset into --- train_A: {}'.format(len(train_A)))
    if type(n_test[0])==float:
        print('                           | --- test_A: {} [ratio: {}]'.format(len(test_A), n_test[0]))
    else:
        print('                           | --- test_A: {}'.format(len(test_A)))
    if type(n_train[1])==float:
        print('                           | --- train_B: {} [ratio: {}]'.format(len(train_B), n_train[1]))
    else:
        print('                           | --- train_B: {}'.format(len(train_B)))
    if type(n_test[1])==float:
        print('                           | --- test_B: {} [ratio: {}]'.format(len(test_B), n_test[1]))
    else:
        print('                           | --- test_B: {}'.format(len(test_B)))
    
def message_model_train_opt(opt):
    # Model-architecture-related
    print('=========================================')
    print('------------------ Model-----------------')
    print('=========================================')
    print('|--- number of channels of input data: input_nc = {}'.format(opt.input_nc))
    print('|--- number of channels of output data: output_nc = {}'.format(opt.output_nc))
    print('--------------Generator------------------')
    print('|--- number of filters in first layer of generator: G_init_filter = {}'.format(opt.G_init_filter))
    print('|--- number of downsampling/upsampling blocks in generator: G_depth = {}'.format(opt.G_depth))
    print('|--- number of residual blocks in generator: G_width = {}'.format(opt.G_width))
    print('------------Discriminator----------------')
    print('|--- number of filters in first layer of discriminator: D_init_filter = {}'.format(opt.D_init_filter))
    print('|--- number of convolutional blocks for discriminator: D_depth = {}'.format(opt.D_depth))
    print('=========================================')
    
    # Training-scheme-related
    print('=========================================')
    print('-------------- Train Options-------------')
    print('=========================================')
    print('|--- number of epochs for training: n_epochs = {}'.format(opt.n_epochs))
    print('|--- epoch from which learning rate decreases to 0 linearly: decay_epoch = {}'.format(opt.decay_epoch))
    print('|--- size of the batches: batchSize = {}'.format(opt.batchSize))
    print('|--- initial learning rate: lr = {}'.format(opt.lr))
    print('|--- hyperparameter controlling the relative importance of the losses: lambda_= {}'.format(opt.lambda_))
    print('|--- add identity loss in cost function: identity_loss = {}'.format(opt.identity_loss))
    print('|--- size of buffer used to train discriminator: buffer_size = {}'.format(opt.buffer_size))
    
    print('|--- use GPU computation: cuda = {}'.format(opt.cuda))
    print('|--- number of cpu threads to use during batch generation: n_cpu = {}'.format(opt.n_cpu))
    print('|--- frequency of saving model\'s parameters: frequency = {}'.format(opt.frequency))

    print('=========================================')
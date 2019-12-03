Quick overwiew over:
1. Dataset.py and load_data fucntion
2. data_visualization
3. train.py

================================================================================================================================
1.
Dataset.py contains load_data funtcion

load_data(dataset, name_model, root='.', mode='load_generator', reuse=None, opt=None, n_train=[0.9, 0.9], n_test=[0.1, 0.1], transformation=['resize', 'crop', 'flip', 'normalize'], load_size=286, crop_size=256)

Example: 
>>> from dataset import load_data
>>> generator_train, generator_test = load_data("landscape2monet", "model_test")
#then to plot the images:
>>> im_train = generator_train() #call the generator to get the images
>>> im_test = generator_test() #call the generator to get the images
>>> imshow(im_train[0]) #to show train image of class A
>>> imshow(im_train[1]) #to show train image of class B
>>> imshow(im_test[0]) #to show test image of class A
>>> imshow(im_test[1]) #to show test image of class B

----> Help on function load_data in module dataset:

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
        'load_generator' | 'load_dataset'. 
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


----> Returns in case of:
-------- mode: "load_generator": 
        generaor_im_train (class Generator)       -- a generator of train images. When called will return a list of the form [image from train set of "A", image from train set of "B]
        generator_im_test (class Generator)       -- a generator of test images. When called will return a list of the form [image from test set of "A", image from test set of "B]

-------- mode: "load_dataset":
        dataset (class DataLoader)       -- a dataset to use for the training phase
    """
===================================================================================================================================
2.
data_visualization.py

shows a plot of list of images, and if outfile is chosen, will save the figure in a specified extension and uotput folder.

def data_visualization(list_images, outfile = False, output_path = '.', name_image_to_save = "test", dpi_to_use = 300, extension = ".png", number_of_columns = 5, figure_size = [20,15])


----> Help on function data_visualization in module data_visualization:


        """shows a plot of list of images, and if outfile is chosen, will save a the figure in a specified extension and uotput folder.
        Parameters:
            list_images [list][required]         -- original option parser
            outfile [bool][optional]             -- True to save the images, Flase to only show the plots
            output_path [str][optional]          -- the directory where to save the image
            name_image_to_save [str][optional]   -- given name to the image to save
            dpi_to_use [int][optional]           -- dpi to use for the image
            extension [str][optional]            -- extension for the image to save ('.png', '.jpg', '.jpeg')
            number_of_columns [int][optional]    -- the number of columns to use for the subplots
            figure_size [list][optional]         -- figure size of the form [int, int]
            
        Returns:
            plot a figure of the given images
            if outifle == True:
                save teh figure with the given extension and dpi to the specified output folder.
        """
===================================================================================================================================
3.
train.py

script to train the network.

examples: 
python train.py
python train.py --dataset 'horse2zebra' --n_train_A 100 n_test_A 5 n_train_B 50 n_test_B 2 --cuda
python train.py --dataset 'landscape2vangogh' --model "test_landscape2VG" --n_train_A 100 n_test_A 5 n_train_B 0.8 n_test_B 0.2 --cuda --frequency 10

list of options:

--dataroot [type=str][default='/home/space/datasets/Unpaired-I2I-translation'] --- root directory of dataset
--dataset [type=str][default='landscape2monet'] --- dataset to use for the training
--folder_A [type=str][default='A'] --- name of directory for dataset A in dataroot
--folder_B [type=str][default='B'] --- name of directory for dataset B in dataroot
--n_train_A [type=int or float][default=0.9] --- number of train samples from dataset A. If n >= 1.0, then the specified number will be converted to int and used as exact number of samples to be used
--n_test_A [type=int or float][default=0.1] --- number of test samples from dataset A. If n >= 1.0, then the specified number will be converted to int and used as exact number of samples to be used
--n_train_B [type=int or float][default=0.9] --- number of train samples from dataset B. If n >= 1.0, then the specified number will be converted to int and used as exact number of samples to be used
--n_test_B [type=int or float][default=0.1] --- number of test samples from dataset B. If n >= 1.0, then tthe specified number will be converted to int and used as exact number of samples to be used
--model [type=str][default='model'] --- name of model trained
--reuse_model [type=str][default=None] --- name of the model, which we want to use its saved splitted datasets
--resize_size [type=int][default=286] --- resize of training images to <resize_size x resize_size>
--crop_size [type=int][default=256] --- crop the image to <crop_size x crop_size>
--BW [action='store_false'] --- Black&White/Grayscale images removal (if --BW is used then there will be no grayscale images removal)
--resize [action='store_false'] --- resize the images to a given value (if --resize is used then there will be no resize transformation)
--crop [action='store_false'] --- random crop the images to a given value (if --crop is used then there will be no crop transformation)
--flip [action='store_false] --- random horizontal flip (if --flip is used then there will be no flip transformation)
--normalize [action='store_false'] --- normalize the images (if --normalize is used then there will be no normalization transformation)

--G_init_filter [type=int][default=64] --- number of filters in first layer of generator
--G_depth [type=int][default=2] --- number of downsampling/upsampling blocks in generator
--G_width [type=int][default=9] --- number of residual blocks in generator
--D_init_filter [type=int][default=64] --- number of filters in first layer of discriminator
--D_depth [type=int][default=4] --- number of convolutional blocks for discriminator
--input_nc [type=int][default=3] --- number of channels of input data
--output_nc [type=int][default=3] --- number of channels of output data

--n_epochs [type=int][default=200] --- number of epochs for training
--decay_epoch [type=int][default=100] --- epoch from which learning rate decreases to 0 linearly
--batchSize [type=int][default=1] --- size of the batches
--lr [type=float][default=0.0002] --- initial learning rate')
--lambda_ [type=float][default=10.0] --- hyperparameter controlling the relative importance of the losses
--identity_loss [action='store_false] --- add identity loss in cost function
--buffer_size [type=int, default=50] --- size of buffer used to train discriminator

--cuda [action='store_true'] --- use GPU computation
--n_cpu [type=int][default=8] --- number of cpu threads to use during batch generation
--frequency [type=int][default=5] --- frequency of saving models parameters


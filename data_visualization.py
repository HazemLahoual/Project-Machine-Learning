import os
import matplotlib.pyplot as plt

def data_visualization(list_images, outfile = False, output_path = '.', name_image_to_save = "test", dpi_to_use = 300, extension = ".png", number_of_columns = 5, figure_size = [20,15]):
    
    """
        shows a plot of list of images, and if outfile is chosen, will save the figure in a specified extension and uotput folder.
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
                save teh figure with the given extension and dpi to the specified output folder
    """
        
    number_images = len(list_images)
    number_of_rows = number_images // number_of_columns 
    number_of_rows += number_images % number_of_columns
    
    position = range(1, number_images+1)
    fig = plt.figure(figsize = figure_size)
    
    for i in range(number_images):
        ax = fig.add_subplot(number_of_rows, number_of_columns, position[i])
        ax.imshow(list_images[i], aspect="auto")
        plt.axis('off')

    if outfile:
        save_to = os.path.join(output_path, name_im_to_save)
        fig.savefig(save_to+extension, dpi=dpi_to_use, bbox_inches = 'tight', pad_inches = 0.1)
    
    
    plt.show() 
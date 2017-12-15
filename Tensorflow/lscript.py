'''
Helper functions for:
Items {
1:'viewing single image'
2:'viewing multiple images'
3:''}

Author: @leopauly
'''

# setting seeds
from numpy.random import seed
seed(1)
import os
os.environ['PYTHONHASHSEED'] = '2'
import tensorflow as tf
tf.set_random_seed(3)

#Imports
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

def view_image(img,image_details=False):
    '''Method for displaying a single image'''
    if (image_details):
        print('shape of image: {}, Type of image {}: '.format(np.shape(img),img.dtype))
        print('Image array \n:',img)
    #plt.figure(figsize=(20,20))
    plt.imshow(img)
    plt.gray()
    plt.show()
    
def view_images(img,labels,axis_show='off'):
    ''' Displaying multiple images as subplots '''
    plt.figure(figsize=(10,10))#figsize=(16,16))
    for i,_ in enumerate(img):
            plt.subplot(3,20,i+1)
            plt.imshow(img[i])
            plt.axis(axis_show)
            plt.title(str(labels[i]))
    plt.gray()
    plt.show()
    
def reshape_grayscale_as_tensor(batch_x):
    ''' reshape numpy grayscale image arrays into tensor format'''
    batch_x = batch_x.reshape(batch_x.shape[0],batch_x.shape[1], batch_x.shape[2],1)
    return batch_x

def reshape_rgb_as_tensor(batch_x):
    ''' reshape numpy grayscale image arrays into tensor format'''
    batch_x = batch_x.reshape(batch_x.shape[0],batch_x.shape[1], batch_x.shape[2],3)
    return batch_x

def plot_values_with_legends(x,y,legend_to_plot,x_axis,y_axis,title,color='red'):
    patch = mpatches.Patch(color=color, label=legend_to_plot)
    plt.figure(figsize=(5,5))
    plt.plot(x,y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.legend(handles=[patch])
    plt.show()

    
    


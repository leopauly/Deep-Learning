'''
Script containing functions related to dataset preparation

Author: @leopauly
'''

# setting seeds
from numpy.random import seed
seed(1)
import os
os.environ['PYTHONHASHSEED'] = '2'
import tensorflow as tf
tf.set_random_seed(3)

# Imports 
import numpy as np

def conv_net(x, img_rows,img_cols,keep_prob):

    # Convolution Layer 1,2 and Maxpooling Layer 12
    conv1 = tf.layers.conv2d(inputs=x,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    pool12 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    

    # Convolution Layer 3,4 and Maxpooling Layer 34
    conv3 = tf.layers.conv2d(inputs=pool12,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    pool34 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    flat = tf.reshape(pool34, [-1, 4 * 4 * 64]) 
    
    dense1 = tf.layers.dense(inputs=flat, units=4 * 4 * 64, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=keep_prob)
    dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=keep_prob)
    
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=10)
    
    # Reading summaries
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('flat', flat)
    tf.summary.histogram('pool12', pool12)
    
    return logits



def create_summaries():
    return 0

# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxW[x3] image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data
    

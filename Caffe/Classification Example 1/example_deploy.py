#Deployment script

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

#setting up the path to caffe_root
import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

#setting up the GPU
caffe.set_device(0)
caffe.set_mode_gpu()

model_file = './example_deploy.prototxt'
pretrained_weights = './logdir/dnn_iter_200000.caffemodel'
net= caffe.Net (model_file,pretrained_weights, caffe.TEST)

im = np.array(Image.open('./images/20455.jpg'))
im=im/255
im = np.reshape(im,(3,32,32))
im_input = im[np.newaxis,:,:,:]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input
prediction = net.forward()
print('Probability of image being human:',prediction['prob'][0][0])
print('Probability of image being animal:',prediction['prob'][0][1])

## Simple forward pass
#net.forward({input_image })
## Printing output of different layers
#print('forward')
#print(net.blobs['conv1'].data.shape)
#print(net.blobs['conv1'].data)

## Simple backward pass
#net.backward()
## Printing output of different layers (the gradients stored while performing backward pass)
#print('backward')
#print(net.blobs['conv1'].diff.shape)
#print(net.blobs['conv1'].diff)



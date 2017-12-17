# A simple CNN sample progam written using caffe framework using python interface.
# Classification example used here is that of a toy problem of classifying images into human or animal images (toy dataset used)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#setting up the path to caffe_root
import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

#setting up the GPU
caffe.set_device(0)
caffe.set_mode_gpu()

#net= caffe.Net (example_train_test.prototxt, caffe.TRAIN)
solver = caffe.SGDSolver('solver.prototxt')
solver.step(1000001)
#print(solver.net.blobs['ip2'].data.shape)
#print(solver.net.blobs['ip2'].data)
#print(solver.net.blobs['loss'].data.shape)
#print(solver.net.blobs['loss'].data)

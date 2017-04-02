
# coding: utf-8

# In[1]:

# My implementation of VGGNet
# Implemented by leopauly: cnlp@leeds.ac.uk

import numpy
import datetime
import glob
from PIL import Image  
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau

K.set_image_dim_ordering('th')


# In[2]:

seed = 7
numpy.random.seed(seed)


# In[3]:

num_classes=10
channel=3
img_rows=32
img_cols=32


# In[4]:

# Prepare Dataset

(x_train, y_train),(x_test,y_test)=cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], channel,img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], channel,img_rows, img_cols)
input_shape=(3,img_rows,img_cols)

x_test=x_test.astype('float32')
x_train=x_train.astype('float32')
x_test=x_test/255.
x_train=x_train/255.

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# In[5]:

#Define model

vgg= Sequential()
vgg.add (Convolution2D(64, 3, 3, input_shape=input_shape, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005)))
vgg.add (Convolution2D(64, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add(MaxPooling2D((2,2), strides=(2,2),border_mode='valid', dim_ordering='th'))

vgg.add (Convolution2D(128, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add (Convolution2D(128, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add(MaxPooling2D((2,2), strides=(2,2),border_mode='valid', dim_ordering='th'))

vgg.add (Convolution2D(256, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add (Convolution2D(256, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add (Convolution2D(256, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add(MaxPooling2D((2,2), strides=(2,2),border_mode='valid', dim_ordering='th'))

vgg.add (Convolution2D(512, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add (Convolution2D(512, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add (Convolution2D(512, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add(MaxPooling2D((2,2), strides=(2,2),border_mode='valid', dim_ordering='th'))

vgg.add (Convolution2D(512, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add (Convolution2D(512, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add (Convolution2D(512, 3, 3, activation='relu',border_mode='same', init='glorot_normal',W_regularizer=l2(0.0005), ))
vgg.add(MaxPooling2D((2,2), strides=(2,2),border_mode='valid', dim_ordering='th'))

vgg.add(Flatten())
vgg.add(Dense(4096, activation='relu'))
vgg.add(Dropout(0.5))
vgg.add(Dense(4096, activation='relu'))
vgg.add(Dropout(0.5))
vgg.add(Dense(num_classes, activation='softmax'))



# In[ ]:

#Compile model

epochs = 100
lrate = 0.01
sgd = SGD(lr=lrate, momentum=0.9, nesterov=True)
vgg.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(vgg.summary())


# In[ ]:

# printing the time when the training starts
print(datetime.datetime.now())

b_size= 256
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, min_lr=0.0001, epsilon=0.001)
# Fit the model
vgg.fit(x_train, y_train, validation_split=0.2, nb_epoch=epochs, batch_size=b_size, callbacks=[reduce_lr], shuffle='batch')

# printing the time when the training finishes
print(datetime.datetime.now())


# In[ ]:

scores = vgg.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:




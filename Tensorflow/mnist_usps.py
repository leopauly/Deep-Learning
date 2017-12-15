'''
Script containing functions related to dataset preparation

Author: @leopauly
'''

# Imports
import scipy.misc as misc
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata, load_iris, load_digits
import scipy.misc
import numpy as np

def dataset(normalisation=False,store=False,m=.1,n=.1):
    #Tag: MNIST
    mnist = fetch_mldata("MNIST original")
    mnist_x=mnist.data
    
    mnist_x=np.reshape(mnist_x[:],[mnist_x.shape[0],28,28])
    mnist_x_new=np.zeros([70000,16,16])
    for i in range(mnist_x.shape[0]):
        mnist_x_new[i,:,:]=misc.imresize(mnist_x[i],[16,16])
        
    mnist_x_new = mnist_x_new.astype('float32')

    #Tag: USPS
    usps = fetch_mldata("USPS")
    usps_x=usps.data
        
    usps_x_new=np.zeros([9298,16,16])
    usps_x_new=np.reshape(usps_x[:],[usps_x.shape[0],16,16])
    usps_x_new=(usps_x_new-(-1))/2
    usps_x_new = usps_x_new.astype('float32')
    usps_x_new=usps_x_new*255

    # if store==True then the images will be written into the current folder
    if (store==True):
        for i in range (usps_x_new.shape[0]):
            print(i)
            c=str(i)
            scipy.misc.toimage(usps_x_new[i], cmin=0.0, cmax=255).save('./data/usps/usps'+c+'.png')
        print('Stored USPS data in the current directory')

        for i in range (mnist_x_new.shape[0]):
            c=str(i)
            print(i)
            scipy.misc.toimage(mnist_x_new[i], cmin=0.0, cmax=255).save('./data/mnist/mnist'+c+'.png')
        print('Stored MNIST data in the current directory')

    # Normalises data if normalisation arguement is true
    if (normalisation==True):
        mnist_x_new=mnist_x_new/255
        usps_x_new=usps_x_new/255

    # Spliting data into validation/testing and training data
    mnist_x_new_train, mnist_x_new_test, mnist_y_new_train, mnist_y_new_test = train_test_split(mnist_x_new, mnist.target,test_size=m,random_state=0)
    usps_x_new_train, usps_x_new_test, usps_y_new_train, usps_y_new_test = train_test_split(usps_x_new, usps.target,test_size=n,random_state=0)
    
    return mnist_x_new_train, mnist_x_new_test, mnist_y_new_train, mnist_y_new_test,usps_x_new_train, usps_x_new_test, usps_y_new_train-1, usps_y_new_test-1
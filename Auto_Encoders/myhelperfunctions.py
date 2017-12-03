import numpy as np
import scipy.misc as misc
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

#Loading data from otther sources
def datasets(m=.1,n=.1):
    #Preparing data MNIST and USPS
    from sklearn.datasets import fetch_mldata, load_iris, load_digits
    usps = fetch_mldata("USPS")
    mnist = fetch_mldata("MNIST original")

    mnist_x=mnist.data
    usps_x=usps.data

    mnist_x=np.reshape(mnist_x[:],[mnist_x.shape[0],28,28])
    mnist_x_new=np.zeros([70000,16,16])
    for i in range(mnist_x.shape[0]):
        mnist_x_new[i,:,:]=misc.imresize(mnist_x[i],[16,16])

    usps_x_new=np.zeros([9298,16,16])

    #Tag: MNIST
    mnist_x_new=np.reshape(mnist_x_new[:],[mnist_x_new.shape[0],256])
    mnist_x_new=mnist_x_new/255
    mnist_x_new = mnist_x_new.astype('float32')

    #Tag: USPS
    usps_x_new=usps_x
    usps_x_new=(usps_x_new-(-1))/2
    usps_x_new = usps_x_new.astype('float32')
   
    #Display numpy-images : MNIST 16x16 after normalisation to 0-1
    print('Display numpy-images : MNIST 16x16 after normalisation to 0-1')
    display=np.reshape(mnist_x_new[1],[16,16])
    print(np.shape(display))
    print(display)
    plt.imshow(display)
    plt.gray()
    plt.show()
    display.dtype


    #Display numpy-images : USPS 16x16 after normalisation to 0-1
    print('Display numpy-images : USPS 16x16 after normalisation to 0-1')
    display=np.reshape(usps_x_new[1],[16,16])
    print(np.shape(display))
    print(display)
    plt.imshow(display)
    plt.gray()
    plt.show()
    display.dtype

    #Split into Test and train data
    from sklearn.model_selection import train_test_split
    mnist_x_new_train, mnist_x_new_test, mnist_y_new_train, mnist_y_new_test = train_test_split(mnist_x_new, mnist.target,test_size=m,random_state=0)
    usps_x_new_train, usps_x_new_test, usps_y_new_train, usps_y_new_test = train_test_split(usps_x_new, usps.target,test_size=n,random_state=0)

    print(np.shape(mnist_x_new_test))
    print(np.shape(usps_x_new_test))

    print(np.shape(mnist_y_new_test))
    print(np.shape(usps_y_new_test))
    
    return mnist_x_new_train, mnist_x_new_test, mnist_y_new_train, mnist_y_new_test,usps_x_new_train, usps_x_new_test, usps_y_new_train, usps_y_new_test
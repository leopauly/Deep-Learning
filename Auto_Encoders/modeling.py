from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D, Input, Lambda,UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Model
from keras import backend as K
K.set_image_dim_ordering('th')

import keras

class models:
    @staticmethod
    def build_lenet(channel,width, height,last_activation):
        model = Sequential()
        model.add(Convolution2D(32,3,3,activation='relu',input_shape=(channel, width,height), border_mode="same"))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Convolution2D(64,3,3, activation='relu', border_mode="same"))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        #model.add(Dropout(0.25))
        model.add(Activation(last_activation))
        return model
    
    def build_lenet_2(channel,width, height,last_activation):
        model = Sequential()
        model.add(Convolution2D(32,3,3,activation='relu',input_shape=(channel, width,height), border_mode="same"))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Convolution2D(64,3,3, activation='relu', border_mode="same"))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Convolution2D(64,4,4,activation='relu', border_mode="valid"))
        #model.add(Dropout(0.25))
        model.add(Activation(last_activation))
        return model
    
    
    def build_classifier(inputs, classes):
            inputs_c = Input(shape=(inputs,))
            #layer1= Dense(512, activation='relu')(inputs_c)
            #layer2= Dense(256, activation='relu')(layer1)
            #layer3= Dense(128, activation='relu')(layer2)
            output = Dense(classes, activation='softmax')(inputs_c)
            classifier=Model(inputs_c,output)
            return classifier
        
    def build_classifier_2(inputs, classes):
            inputs_c = Input(shape=(inputs,))
            layer1= Dense(512, activation='relu')(inputs_c)
            layer2= Dense(256, activation='relu')(layer1)
            layer3= Dense(128, activation='relu')(layer2)
            output = Dense(classes, activation='softmax')(layer3)
            classifier=Model(inputs_c,output)
            return classifier
    
    def build_classifier_3(inputs, classes):
            inputs_c = Input(shape=(inputs,))
            layer1= Dense(512, activation='relu')(inputs_c)
            layer2= Dense(256, activation='relu')(layer1)
            layer3= Dense(128, activation='relu')(layer2)
            layer4= Dense(64, activation='relu')(layer3)
            output = Dense(classes, activation='softmax')(layer4)
            classifier=Model(inputs_c,output)
            return classifier
        
    def build_decoder(c,w,h):
        
        model_d = Sequential()
        model_d.add(Convolution2D(64,3,3,activation='relu',input_shape=(c,w,h), border_mode="same"))
        model_d.add(UpSampling2D((2,2)))
        model_d.add(Convolution2D(32,3,3, activation='relu', border_mode="same"))
        model_d.add(UpSampling2D((2,2)))
        model_d.add(Convolution2D(1,3,3, activation='sigmoid', border_mode="same"))
        #model.add(Dropout(0.25))
        return model_d
    
    def build_decoder_2(c,w,h):
        
        model_d = Sequential()
        model_d.add(Convolution2D(64,3,3,activation='relu',input_shape=(c,w,h), border_mode="same"))
        model_d.add(UpSampling2D((4,4)))
        model_d.add(Convolution2D(64,3,3,activation='relu',input_shape=(c,w,h), border_mode="same"))
        model_d.add(UpSampling2D((2,2)))
        model_d.add(Convolution2D(32,3,3, activation='relu', border_mode="same"))
        model_d.add(UpSampling2D((2,2)))
        model_d.add(Convolution2D(1,3,3, activation='sigmoid', border_mode="same"))
        #model.add(Dropout(0.25))
        return model_d
        
 
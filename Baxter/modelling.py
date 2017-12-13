def modelC3D_theano(load_weights=True,summary=True):
    '''
    '''
    ## Imports
    import h5py
    from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Sequential
    
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv1',
            subsample=(1, 1, 1),
            input_shape=(3, 16, 112, 112),
            trainable=False))
    model.add(MaxPooling3D(pool_size=(1, 2, 2),strides=(1, 2, 2),border_mode='valid',name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128,3,3,3,activation='relu',border_mode='same',name='conv2',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv3a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(Convolution3D(256,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv3b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv4a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv4b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv5a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv5b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropadding'))
    model.add(MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool5'))
    model.add(Flatten(name='flatten'))
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5, name='do2'))
    model.add(Dense(487, activation='softmax', name='fc8'))

    # Load weights
    if load_weights:
        model.load_weights('/media/leo/PENDRIVELEO/s2l/c3d-sports1M_weights.h5')
        print('weighs loaded')
    if summary:
        print(model.summary())
        
    return model

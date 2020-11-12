import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import multiprocessing as mp
import glob
import os, cv2
import random
import ray
import json
import cfg

RESUME = False
NCLASS = cfg.NCLASS
MY_BATCH = 256

@ray.remote
def load_im(fname,idx):
    print(fname)
    im = cv2.imread(fname)
    im = (im - 128)/256
    im = cv2.resize(im, (227,227), interpolation=cv2.INTER_AREA)
    return im,idx

if __name__=='__main__':
    ray.init(num_cpus=8)

    np.random.seed(1000)

    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(NCLASS, activation='softmax')
    ])

    model.summary()

    # Compile the model
    #model.compile(loss=keras.losses.categorical_crossentropy, 
    #        optimizer='adam', 
    #        metrics=['accuracy'])

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
    if RESUME:
        model.load_weights('alexnet.weights')

    with open('train_set.json') as f:
        train_ls = json.load(f)


    for m in range(10):
        new_train_ls = train_ls[:]
        random.shuffle(new_train_ls)
        train_ls_quad = []
        try:
            while True:
                ls1 = []
                for n in range(MY_BATCH):
                    ls1.append(new_train_ls.pop(0))
                train_ls_quad.append(ls1)
        except Exception as e:
            pass
        
        for item in train_ls_quad:
            x_train = np.zeros((MY_BATCH,227,227,3))
            y_train = np.zeros((MY_BATCH,NCLASS))
            USE_RAY = True
            if USE_RAY:
                pid_ls = []
                for n, sample in enumerate(item):
                    idx = sample[0]
                    fname = sample[1]
                    pid_ls.append(load_im.remote(fname,idx))
                for n in range(len(pid_ls)):
                    result = ray.get([pid_ls[n]])
                    result = result[0]
                    x_train[n,:,:,:] = result[0].copy()
                    idx = result[1]
                    y_train[n,idx] = 1
                model.fit(x_train, y_train, batch_size=8, epochs=50)
                model.save_weights('alexnet.weights')
            else:
                for n, sample in enumerate(item):
                    idx = sample[0]
                    fname = sample[1]
                    print(fname, '   ', idx, m)
                    im = cv2.imread(fname).astype(np.float)
                    im = (im - 128)/256
                    im = cv2.resize(im, (227,227), interpolation=cv2.INTER_AREA)
                    x_train[n,:,:,:] = im
                    y_train[n,idx] = 1
                model.fit(x_train, y_train, batch_size=16, epochs=50)
                model.save_weights('alexnet.weights')

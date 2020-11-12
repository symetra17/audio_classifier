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
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import cfg
import json
NCLASS = cfg.NCLASS
CNN_SIZE = 227
inp = R'C:\Users\dva\Documents\audio_design\images'
inp_folder = R'my_samples\*.wav'
word_list = ['down','five','four','left','no','off','on','right','six','three','up','yes']
accuracy_stats = False

def build_ResNet50(input_tensor_shape):
    base_model = ResNet50(include_top=False, input_shape=input_tensor_shape)
    x_model = base_model.output
    x_model = tf.keras.layers.GlobalAveragePooling2D()(x_model)
    x_model = tf.keras.layers.Dense(1024, activation='relu',name='fc1_Dense')(x_model)
    x_model = tf.keras.layers.Dropout(0.5, name='dropout_1')(x_model)
    x_model = tf.keras.layers.Dense(256, activation='relu',name='fc2_Dense')(x_model)
    x_model = tf.keras.layers.Dropout(0.5, name='dropout_2')(x_model)
    predictions = tf.keras.layers.Dense(NCLASS, activation='sigmoid',name='output_layer')(x_model)    
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)    
    return model

@ray.remote
def load_im(fname,idx):
    print(fname)
    im = cv2.imread(fname)
    im = (im - 128)/256
    im = cv2.resize(im, (CNN_SIZE,CNN_SIZE), interpolation=cv2.INTER_AREA)
    return im,idx,fname

if __name__=='__main__':
    ray.init(num_cpus=12)

    np.random.seed(1000)

    model_alexnet = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', 
                    input_shape=(CNN_SIZE,CNN_SIZE,3)),
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
    #model = build_ResNet50((CNN_SIZE,CNN_SIZE,3))
    model = model_alexnet

    model.load_weights('alexnet.weights')

    with open('test_set.json') as f:
        predict_ls = json.load(f)

    import generate_spectral
    files = glob.glob(inp_folder)
    predict_ls = []
    for fname in files:
        outname = os.path.splitext(fname)[0] + '.png'
        generate_spectral.get_feature(fname, outname)
        predict_ls.append((0,outname))

    random.shuffle(predict_ls)

    correct = 0
    incorrect = 0
    for pair in predict_ls:
        idx = pair[0]
        fname = pair[1]
        im = cv2.imread(fname)
        im = (im - 128)/256
        im = cv2.resize(im, (CNN_SIZE,CNN_SIZE), interpolation=cv2.INTER_AREA)
        im2 = np.zeros((1,CNN_SIZE,CNN_SIZE,3))
        im2[0,:,:,:] = im        
        y = model.predict(im2)
        idx_pred = np.argmax(y[0])
        #print(fname, idx_pred, '  ground true',idx, '%.2f'%y[0].max())
        print(fname, '  predicted:', word_list[idx_pred])
        if idx_pred == idx:
            correct += 1
        else:
            incorrect += 1

    if accuracy_stats:
        print('Correct:', correct, '  Incorrect:', incorrect)
        print('Accuracy: %.1f%%'%(100*correct/(correct+incorrect)))

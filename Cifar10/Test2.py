from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.models import load_model
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
import matplotlib.pyplot as plt
from keras.models import load_model


class test():
    def __init__(self):

        (self.x_train_raw, self.y_train_raw), (self.x_test_raw, self.y_test_raw) = cifar10.load_data()
        self.label_dict={0:"airplain",1:"car",2:"bird",3:"cat",4:"deer",5:"dog",
                6:"frog",7:"horse",8:"ship",9:"truck"}

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.x_train = self.x_train_raw.astype('float32')
        self.x_test = self.x_test_raw.astype('float32')

        self.y_train = keras.utils.to_categorical(self.y_train_raw, 10)
        self.y_test = keras.utils.to_categorical(self.y_test_raw, 10)
        self.model = load_model('vgg16.h5')
        
        self.predicted_x = self.model.predict(self.x_test)
    def normalize_production(self,x):
        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)
    
    def ShowImg(self):
        fig=plt.figure(figsize=(12, 6))
        columns, rows = 5, 2
        for i in range(1, columns*rows +1):
            randIdx = random.randint(0, 49999)
            img = self.x_train_raw[randIdx]
            ax = fig.add_subplot(rows, columns, i)
            ax.title.set_text(self.label_dict[int(self.y_train_raw[randIdx])])
            plt.imshow(img)
        plt.show()
    
    def inference(self, imgIdx):
        #imgIdx = 5
        f, axarr = plt.subplots(2,1) 

        k = self.label_dict.values()
        v = self.predicted_x[imgIdx]
        axarr[1].bar(k, v, width = 0.5)

        img = self.x_test_raw[imgIdx]
        axarr[0].imshow(img)
        plt.show()
    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)
   
        
    def summary(self):
        self.model.summary()

        
    


#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
from PyQt5 import QtWidgets, QtGui, QtCore
from HW import Ui_MainWindow
import sys
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
import matplotlib.pyplot as plt
from keras.models import load_model
from Test2 import test


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
       
    
def Paremeters():
    print('learning is 0.1')
    print('batch_size is 128')
    print('Optimizer is SGD')
    
def Show():
    img = cv2.imread('Line.png')
    cv2.namedWindow("output") 
    cv2.imshow('output', img)

    
def Window():
    if __name__ == '__main__':
        app = QtWidgets.QApplication([])
        window = MainWindow() #Declare
        spinBox = window.ui.spinBox
        Rm = window.ui.pushButton
        Pa= window.ui.pushButton_2
        St = window.ui.pushButton_3
        Acc = window.ui.pushButton_4
        Test = window.ui.pushButton_5
        t = test()
        ## btns onClicked
        ### 5.1
        Rm.clicked.connect(lambda :t.ShowImg())
        ### 5.2
        Pa.clicked.connect(lambda :Paremeters())
        ### 5.3
        St.clicked.connect(lambda :t.summary())
        ### 5.4
        Acc.clicked.connect(lambda :Show())
        ### 5.5
        Test.clicked.connect(lambda :t.inference(spinBox.value()))
        #Test.clicked.connect(lambda :print(spinBox.value()))
        window.show()
        sys.exit(app.exec_())
Window()




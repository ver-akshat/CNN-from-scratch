# main class for running CNN over MNISt dataset

import numpy as np
from utils import *
import tensorflow as tf

def main():
    # load training data
    (X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
    X_train=X_train[:5000]
    y_train=y_train[:5000]
    # defining network
    layers=[ConvolutionLayer(16,3),MaxPoolingLayer(2),SoftmaxLayer(13*13*16,10)]
    for epoch in range(7):
        print('Epoch {}->'.format(epoch+1))
        # shuffle training data
        permutation=np.random.permutation(len(X_train))
        X_train=X_train[permutation]
        y_train=y_train[permutation]
        #training CNN
        loss=0
        accuracy=0
        for i,(image,label) in enumerate(zip(X_train,y_train)):
            if i % 100 == 0: # Every 100 examples
                print("Step {}. For the last 100 steps: average loss {}, accuracy {}".format(i+1,loss/100,accuracy))
                loss=0
                accuracy=0
            loss_1,accuracy_1=CNN_training(image,label,layers)
            loss+=loss_1
            accuracy+=accuracy_1

if __name__=='__main__':
    main()
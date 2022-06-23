# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:26:53 2022

@author: pc
"""

import numpy as np 
import matplotlib.pyplot as plt

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Bidirectional,Embedding


class ModelCreation():
    def __init__(self):
        pass
    
    def embed_lstm_layer(self,x_train,num_node=128,drop_rate=0.3,
                          output_node=5,embedding_dim=64,vocab_size=10000):
       model = Sequential()
       model.add(Input(shape=(np.shape(x_train)[1])))
       model.add(Embedding(vocab_size, embedding_dim))
       model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
       model.add(Dropout(0.3))
       model.add(LSTM(128))
       model.add(Dropout(0.3))
       model.add(Dense(128,activation='relu'))
       model.add(Dropout(0.3))
       model.add(Dense(output_node,'softmax'))
       model.summary()
        
       return model


class Model_Evaluation():
    def plot_hist_graph(self,hist):
        plt.figure()
        plt.plot(hist.history['loss'],label='Training loss')
        plt.plot(hist.history['val_loss'],label='Validation loss')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(hist.history['acc'],label='Training acc')
        plt.plot(hist.history['val_acc'],label='Validation acc')
        plt.legend()
        plt.show()
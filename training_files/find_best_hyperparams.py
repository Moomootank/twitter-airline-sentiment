# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:01:48 2017

@author: schia

Script that finds the best hyperparameters for the various models used.
"""
import sys
sys.path.append(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files")
from Baseline_LSTM import LSTM_Model

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def load_obj(name):
    with open(name, 'rb') as file:
        return pickle.load(file)

if __name__ == "__main__":
    #Load the embeddings 
    embedding_matrix = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\embedding_matrix.pickle")
    
    tweets_url = r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\tweets_clean.csv"
    tweets_data = pd.read_csv(tweets_url,encoding = "windows-1252") #Apparently windows 1252 removes the error?...
    
    padded_indices = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\padded_indices.pickle")
    labels = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\labels.pickle")
    
    #Let's make sure the thing trains first
    with tf.Graph().as_default():
        model = LSTM_Model()
        model.define_fixed_hyperparams(14640,100,3,732, 1500, 1e-4, 150, embedding_matrix)
        #n_samples, n_features, n_classes, batch, n_epochs, lr, max_l, embeddings
        model.define_network_hyperparams(157 ,0.7)
        #n_hidden units, dropout
        
        model.initialize_ops()
        variables_init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(variables_init)
            losses = model.fit(sess, padded_indices, labels)
            print ("Last training loss is", losses[-1])
            sns.tsplot(losses)
        

    
    
    
    



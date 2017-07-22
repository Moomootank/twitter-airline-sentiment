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
    
    train_indices = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\train_data\train_indices.pickle")
    train_labels = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\train_data\train_labels.pickle")
    
    val_indices = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\validation_data\validation_indices.pickle")
    val_labels = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\validation_data\validation_labels.pickle")
    
    #Let's make sure the thing trains first
    with tf.Graph().as_default():
        model = LSTM_Model()
        model.define_fixed_hyperparams(11858,100,3,659, 100, 1e-4, 150, embedding_matrix)
        #n_samples, n_features, n_classes, batch, n_epochs, lr, max_l, embeddings
        model.define_network_hyperparams(157 ,0.7)
        #n_hidden units, dropout
        
        model.initialize_ops()
        variables_init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(variables_init)
            losses = model.fit(sess, train_indices, train_labels)
            sns.tsplot(losses)
            print ("Calculating prediction loss now....")
            other_avg_loss = model.predict_other(sess, val_indices, val_labels)
        

    
    
    
    



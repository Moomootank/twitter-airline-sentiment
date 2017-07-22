# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:01:48 2017

@author: schia

Script that finds the best hyperparameters for the various models used.
"""
import sys
sys.path.append(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files")
from Baseline_LSTM import LSTM_Model
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def save_obj(obj, directory):
    with open(directory,'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as file:
        return pickle.load(file)
    
def hyperopt_wrapper_nn(train_indices, train_labels, other_indices, other_labels):
    
    def train_model(params):
        #Params are the parameters that you want hyperopt to optimize
        with tf.Graph().as_default():
            model = LSTM_Model()
            model.define_fixed_hyperparams(100,3, 847, 659, 850, 1e-4, 150, embedding_matrix)
            #n_features, n_classes, batch, other_batch, n_epochs, lr, max_l, embeddings
            model.define_network_hyperparams(**params)
            #unfold params into the model
            
            model.initialize_ops()
            variables_init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(variables_init)
                losses = model.fit(sess, train_indices, train_labels)
                sns.tsplot(losses)
                print ("Calculating prediction loss now....")
                other_avg_loss = model.predict_other(sess, val_indices, val_labels)         
                return other_avg_loss # return the validation loss
            
    def minimize_this(params):
        
        score = train_model(params)
        return {'loss': score , 'status': STATUS_OK} #Make sure score is positive. Should be if you use tf.nn.softmax_cross_entropy_with_logits
    
    nndict = {'n_hidden_units': hp.quniform('n_hidden_units', 50, 300, 1), 'n_dropout' : hp.uniform('n_dropout',0.1,0.90)}
    trials = Trials()
    best = fmin(fn=minimize_this, space = nndict, algo= tpe.suggest, max_evals = 2, trials= trials)
    
    save_obj(best, r"/training_logs/best_hyperparams.pickle")
    save_obj(trials, r"/training_logs/hyperparam_trials.pickle")
    print ('Best hyperparams:', best)
    return best, trials
        
    

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
    best, trials = hyperopt_wrapper_nn(train_indices, train_labels, val_indices, val_labels)
        

    
    
    
    



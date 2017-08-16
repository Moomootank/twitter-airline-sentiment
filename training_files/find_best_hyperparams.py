# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:01:48 2017

@author: schia

Script that finds the best hyperparameters for the various models used.
"""
import sys
sys.path.append(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files")
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from Baseline_LSTM import LSTM_Model
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


#=====The following methods define ways to save objects=====
def save_obj(obj, directory):
    with open(directory,'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as file:
        return pickle.load(file)
        
#=====The following methods define ways to train the model and predict using it

def train_model(train_indices, train_labels, save_dir, embedding_matrix, **params):
    '''
    Function that trains a tensorflow model with the desired parameters, then checks loss on validation/test set
    
    '''
    print ("Optimizing params:", params)
    graph = tf.Graph()
    with graph.as_default():
        model = LSTM_Model()
        model.define_fixed_hyperparams(200,3, 1694, 659, 350, 1e-4, 35, 2, embedding_matrix)
        #n_features, n_classes, batch, other_batch, n_epochs, lr, max_l, num_layers, embeddings
        model.define_network_hyperparams(**params)
        #unfold params into the model
        
        model.initialize_ops()
        variables_init = tf.global_variables_initializer()
        sess = tf.Session() # Not using "with tf.Session() as sess" as that would close the session outside of the indent
        
        sess.run(variables_init)
        losses = model.fit(sess, train_indices, train_labels)
        sns.tsplot(losses)
        plt.show()
        plt.clf()
        #other_avg_loss = model.predict_other(sess, other_indices, other_labels)  
        print()
        if save_dir is not None:
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_dir)
            print("Model saved in file: {path}".format(path=save_path))
        return graph, sess, model

def predict_using_model(graph, sess, model, other_indices, other_labels, metric):
    '''
    Function that uses a trained model to predict on the hold-out set
    Arguments:
        graph: graph that the session is on
        sess: Current session
        model: Model that has already been trained on training data
        other_indices: x-values for hold out set
        other_labels: y-values for hold out set
        metric: The accuracy metric that you wish to use
    '''
    print ("Calculating prediction loss now using metric:", metric)
    with graph.as_default():
        if metric=="cross_entropy":
            other_avg_loss = model.predict_other(sess, other_indices, other_labels)
        elif metric=="f1_score":
            other_avg_loss = model.predict_f1(sess, other_indices, other_labels)
        else:
            print ("The evaluation metric you chose has not yet been implemented in the model")
            return
                
        sess.close()
        return other_avg_loss


#=====The following methods define ways to tune the model's hyperparameters

def hyperopt_wrapper_nn(train_indices, train_labels, other_indices, other_labels, embeddings, file_directory, load):    
                   
    def minimize_this(params):       
        graph, session, model = train_model(train_indices, train_labels, None, embeddings, **params)
        score = predict_using_model(graph, session, model, other_indices, other_labels, 'cross_entropy')
        return {'loss': score , 'status': STATUS_OK} #Make sure score is positive. Should be if you use tf.nn.softmax_cross_entropy_with_logits
    
    nndict = {'n_hidden_units': [hp.quniform('n_hidden_units_1', 50, 500, 1)], 
              'n_dropout' : [hp.uniform('n_dropout',0.0,0.90)],
              'n_input_dropout': [hp.uniform('n_input_dropout',0.0,0.90)]}
    
    if load:
        print ("Loading Trials object")
        trials = load_obj(file_directory) #If load, then load some previous trials object and use it; start off at a certain point
    else:
        trials = Trials() # Else, start from scratch
    best = fmin(fn=minimize_this, space = nndict, algo= tpe.suggest, max_evals = 60, trials= trials)
    print ("Saving trials object")
    save_obj(trials, file_directory)
    
    save_obj(best, r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_logs\hyp_two_layer_best_hyperparams.pickle")
    print ('Best hyperparams:', best)
    return best, trials

def randomized_search(train_indices, train_labels, other_indices, other_labels, embeddings, file_directory):
    #Randomized hyperparameter search for tensorflow neural network
    #Used instead of hyperopt because hyperopt tends to crash my potato computer (random kernel errors)
    num_evals = 100
    
    for search in range(num_evals):
        n_hidden_units = [np.random.randint(50,500), np.random.randint(50,500)]
        n_dropout = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
        parameters = {'n_hidden_units': n_hidden_units, 'n_dropout': n_dropout}
        graph, session, model = train_model(train_indices, train_labels, None, embeddings, **parameters)
        val_loss = predict_using_model(graph, session, model, other_indices, other_labels, 'cross_entropy')
        
        with open(file_directory, 'a') as file:
            file.writelines("{units} {dropout} {val}\n".format(units=n_hidden_units, dropout=n_dropout, val = val_loss))
       
if __name__ == "__main__":
    #Load the embeddings 
    embedding_matrix = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\embedding_matrix.pickle")

    #tweets_url = r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\tweets_clean.csv"
    #tweets_data = pd.read_csv(tweets_url,encoding = "windows-1252") #Apparently windows 1252 removes the error?...
    
    train_indices = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\train_data\train_indices.pickle")
    train_labels = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\train_data\train_labels.pickle")
    
    val_indices = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\validation_data\validation_indices.pickle")
    val_labels = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\validation_data\validation_labels.pickle")
    
    test_indices = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\test_data\test_indices.pickle")
    test_labels = load_obj(r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_data\test_data\test_labels.pickle")
    #Let's make sure the thing trains first
    hyp_dir = r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_logs\hyperopt_trials.pickle"
    best, trials = hyperopt_wrapper_nn(train_indices, train_labels, val_indices, val_labels, embedding_matrix, hyp_dir, True)
    #random_search_log = r"D:\Data Science\Projects\twitter-airline-sentiment\training_files\training_logs\two_layer_random_search_log.csv"
    #randomized_search(train_indices, train_labels, val_indices, val_labels, embedding_matrix,  random_search_log)
        
    #Evaluating model performance on test set
    '''
    params = {'n_hidden_units': [266, 271], 'n_dropout': [0.071714659443375, 0.1412809617348177]}
    graph, session, model = train_model(train_indices, train_labels, None, embedding_matrix, **params)
    score = predict_using_model(graph, session, model, test_indices, test_labels, "f1_score")
    '''
    

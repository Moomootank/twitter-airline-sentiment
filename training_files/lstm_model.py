# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 10:23:44 2017

@author: Moomootank
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:03:10 2017

@author: Moomootank

This file contains the basic word-embedding LSTM model 
"""

import time
import sys
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#Parent class for the unidirectional_lstm and bidirectional_lstm models

class LSTM_Model():
    def __init__(self , config):
        #config is an object which specifies the fixed parameters of the model
        self.embed_size = config.embed_size
        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.val_size = config.val_size #batch size for validation set
        self.test_size = config.test_size #batch size for test set
        
        self.num_epochs =  config.num_epochs
        self.learning_rate = config.learning_rate
        self.max_length = config.max_length
        self.num_layers = config.num_layers
        self.pretrained_embeddings = config.pretrained_embeddings   
    
    def define_network_hyperparams(self, n_hidden_units, n_dropout, n_input_dropout):
        self.num_hidden_units = n_hidden_units
        self.num_dropout = n_dropout
        self.num_input_dropout = n_input_dropout
    
    #=====The following functions are the helper functions that help the models ops=====
    def add_placeholders(self):
        with tf.name_scope("Data"):
            self.input_placeholder = tf.placeholder(dtype= tf.int32, shape = (None, self.max_length))
            self.labels_placeholder = tf.placeholder(dtype= tf.int32, shape = (None, self.num_classes))
            #Need dropout to be in placeholder format to make it easier to turn it off during prediction
            self.dropout_placeholder = tf.placeholder(dtype = tf.float32, shape = (self.num_layers)) 
            self.input_dropout_placeholder = tf.placeholder(dtype = tf.float32, shape = (self.num_layers))
               
    def create_feed_dict(self, inputs_batch, labels_batch, n_dropout, n_input_dropout):
        feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch, 
                     self.dropout_placeholder: n_dropout, self.input_dropout_placeholder: n_input_dropout}
        return feed_dict
    
    def add_embedding(self):
        with tf.name_scope("Data"):
            embedding = tf.nn.embedding_lookup(params = self.pretrained_embeddings, ids = self.input_placeholder)
        return tf.cast(embedding, dtype = tf.float32)
    
    def find_batch_length(self, embeddings):
        """
        Each tweet has a different length. Find the length of each tweet;
        this length will be fed to dynamic_rnn so that the model knows when to stop processing
        for each tweet
        Arguments:
            embeddings: word embeddings for the tweet
        """
        with tf.name_scope("Tweet_lengths"):
            embeddings_used = tf.sign(tf.reduce_max(tf.abs(embeddings), axis = 2))
            """
            tf.abs: get the absolute value of these embeddings
            tf.reduce_max, Along the third dimension (the word_embeddings), find the maximum,
            remove the third dimension, so the second dimension is now just the maximum
            i.e. (batch_size, num_words, abs_word_embedding) => (batch_size, max_abs_word_embedding)
            tf.sign(x) is =0 if x =0 or x.isnan, = 1 if x > 0
            """
            
            tweet_length = tf.reduce_sum(embeddings_used, axis=1)
            tweet_length = tf.cast(tweet_length, dtype=tf.int32)
        return tweet_length
    
    def add_cells(self):
        raise NotImplementedError("Each Model should re-implement add_cells method")
    
    #=====The following functions define the main operations of the model=====
    def add_prediction_op(self):
        raise NotImplementedError("Each Model should re-implement add_prediction_op method")
    
    def add_loss_op(self, predictions):
        with tf.name_scope("loss_ops"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.labels_placeholder, logits = predictions)
            loss = tf.reduce_mean(loss)
        return loss
    
    def add_training_op(self, loss):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss)      
        return train_op
    
    def initialize_ops(self):
        self.add_placeholders() # Add the placeholders
        self.pred = self.add_prediction_op() #Add the prediction op
        self.loss = self.add_loss_op(self.pred)
        self.train = self.add_training_op(self.loss)
        
    #===== The following functions execute the model =====
    def get_minibatches(self, data, labels, b_size):
        '''
        Helper function that returns a list of tuples of batch_size sized inputs and labels
        Return n_samples/batch size number of batches
        Args:
            data: dataframe with columns of independent variables
            x_cols: list of length n_features of independent column names
            labels: name of label column       
            b_size: size of each batch
            
        Returns: List of tuples where each tuple is (input_batch, label_batch)
        '''
    
        if data.shape[0] < b_size:
            print ("There are fewer samples than the batch size. Returning the data as is.")
            return [(data, labels)]
        
        data_length = len(data)
        assert data_length==len(labels)
        
        p = np.random.permutation(data_length)
        reshuffled_data = data[p]
        reshuffled_labels = labels[p] #These create copies of the array, so the original copies are untouched
        
        num_batches = math.floor(data_length/b_size) #Number of batches, if b_size is not a factor some data will be left out. Ah well.
        batches = []         
        for i in range(num_batches):
            start = i*b_size
            data_sample = reshuffled_data[start:(start + b_size)] # Sample of b size
            labels_sample = reshuffled_labels[start:(start + b_size)]
            batches.append((data_sample, labels_sample))
        return batches
        
    def run_epoch(self, session, data, labels):
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in self.get_minibatches(data, labels, self.batch_size):
            feed = self.create_feed_dict(input_batch, labels_batch, self.num_dropout, self.num_input_dropout)
            _ , batch_loss = session.run([self.train, self.loss], feed_dict = feed) #self.loss will not run two times. This just fetches the value
            n_minibatches += 1
            total_loss += batch_loss
        epoch_average_loss = total_loss/n_minibatches
        return epoch_average_loss
    
    def fit(self, session, data, labels, other_data, other_labels, saver):
        #data, labels: training data and labels
        #other_data, other_labels: the validation/test data and labels
        
        losses = []
        start = time.time() # Start time of the entire model
        previous = start
        
        best_loss = 100 # Initial val score
        
        for epoch in range(self.num_epochs):
            average_loss = self.run_epoch(session, data, labels)
            losses.append(average_loss) #Tracking loss per epoch
            
            if epoch%10==0 or epoch==self.num_epochs-1:
                val_loss = self.predict_other(session, other_data, other_labels, True)
                if val_loss < best_loss:
                    best_loss = val_loss    
                    saver.save(session, "training_logs/checkpoints/current_best.ckpt")
                    print ("New best model saved. Epoch:", epoch)
            
            if epoch % 50 ==0 or epoch==self.num_epochs-1: #-1 due to characteristics of range
            #This if block just prints out the progressand time taken to reach a certain stage
                current_time = time.time()
                duration = current_time - start
                duration_min = math.floor(duration/60)
                duration_sec = duration % 60
                since_last = current_time - previous
                since_last_min = math.floor(since_last/60)
                since_last_sec = since_last % 60 
                print ("Epoch number {e} completed. Time taken since start: {start_min} min {start_sec} s."
                       .format(e = epoch, start_min = duration_min, start_sec = duration_sec))
                print ("Time taken since last checkpoint: {last_min} min {last_sec} s."
                       .format(last_min = since_last_min, last_sec = since_last_sec ))
                print ("Average loss this epoch is:" , average_loss)
                print ()
                previous = current_time #Set the new "last checkpoint" to this one
                
        return losses #Can try to plot how much the loss has gone down
            
    def predict_other(self, session, other_data, other_labels, val, console=False):
        '''
        Use the trained model to predict a batch of non-train data
        Arguments:
            session: The tensorflow session
            data: the val or test padded word_embedding indices
            other_labels: the val or test labels        
        '''
        if val:
            other_size = self.val_size
        else:
            other_size = self.test_size
               
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in self.get_minibatches(other_data,other_labels, other_size):       
            #Need to do this in batches as we might not have enough memory
            #When we predict on the other set, we must ensure that the input and output dropouts are = 0
            feed = self.create_feed_dict(input_batch, labels_batch, [0]*self.num_layers, [0]*self.num_layers) 
            batch_loss = session.run(self.loss, feed_dict = feed) #It will call itself
            total_loss += batch_loss
            n_minibatches+= 1
            #Idk, return total loss, or average over batches?
        average_loss = total_loss/n_minibatches
        if console:
            print("Total other_loss is:", total_loss)
            print ("Average batch other_loss is:", average_loss)
            print()
        return average_loss 
    
    def predict_f1(self, session, other_data, other_labels, val):
        '''
        Use the trained model to predict the f1 score for a batch of non-train data
        '''
        predictions = []
        labels = []
        if val: #Just a toggle for whether we are predicting on validation or test set
            other_size = self.val_size
        else:
            other_size = self.test_size
            
        for input_batch, labels_batch in self.get_minibatches(other_data,other_labels, other_size):
            feed = self.create_feed_dict(input_batch, labels_batch, [0]*self.num_layers, [0]*self.num_layers)
            batch_probs = session.run(tf.nn.softmax(self.pred), feed_dict = feed)
            pred_label = session.run(tf.argmax(batch_probs, axis = 1)) #batch size; each index is a predicted label
            predictions.extend(pred_label)
            labels_squeezed = np.argmax(labels_batch, axis = 1) #Squeeze one hot encoding into 1d labels , to use in f1_score
            labels.extend(labels_squeezed)
            
        score = f1_score(labels, predictions, average = 'macro')
        print ("The macro average f1 score is:",score)
        print ("The confusion matrix is:\n", confusion_matrix(labels, predictions))
        return (score, labels, predictions, batch_probs)
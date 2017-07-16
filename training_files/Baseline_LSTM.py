# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:03:10 2017

@author: Moomootank

This file contains the basic word-embedding LSTM model 
"""

import time
import copy
import sys
import math

import numpy as np
import pandas as pd
import tensorflow as tf

class LSTM_Model():
    def define_fixed_hyperparams(self, n_samples, n_features, n_classes, batch, n_epochs, lr, max_l, embeddings):
        """
        Store information about data hyperparameters
        n_samples = number of rows in data
        n_features = number of columns (independent variables) in data i.e dimension of word embedding
        n_classes = number of classes to predict
        batch = batch size to use in stochastic gradient descent
        n_epochs = number of epochs to train for 
        lr = learning rate    
        max_l = max length that RNN can extend for (i.e. longest sequence to parse)
        embeddings = pretrained embeddings that we will use
        """
        self.sample_size = n_samples    
        self.embed_size = n_features
        self.num_classes = n_classes
        self.batch_size = batch
        self.num_epochs =  n_epochs
        self.learning_rate = lr
        self.max_length = max_l
        self.pretrained_embeddings = embeddings
    
    def define_network_hyperparams(self, n_hidden_units, n_dropout):
        '''
        Store information about parameters that we would most likely want to tune using random search/hyperopt etc.
        n_hidden_units = number of hidden units in a hidden layer
        n_dropout = dropout rate for the lstm units
        '''
        self.num_hidden_units = n_hidden_units
        self.num_dropout = n_dropout
        
    def pad_sequences(data, max_length, zero_embedding_index):
        """
        Ensures that each input-output sequence pair is of length max_length
        Since we are going to be using tf.nn.embedding_lookup, make the last row of the embedding vector
        a vector of zeroes
        
        So, for the embedding lookup index, make the last row a vector of zeroes (so the second last 
        is now the "unk" token)
        
        Arguments:
            data: List of list of embedding-lookup indices. Each index in data is the list of embedding
            lookup indices for that particular tweet/sentence
            
            max_length: max-length that our RNN will run for   
            zero-embedding index: Index of the zero vector in the embedding matrix
        """
        holder = []
        for tweet in data:
            length = len(tweet)
            if length>=max_length:
                holder.append(tweet[:max_length])
            else:
                number_to_extend = max_length-length
                #Creating deep copies to avoid changing the list
                copied_tweet = copy.deepcopy(tweet)
                holder.append(copied_tweet.extend([zero_embedding_index for i in range(number_to_extend)]))
        return holder
        
    def add_placeholders(self):
        """
        Generates placeholder variables to represent the input tensors.
    
        These placeholders are used as inputs by the rest of the model building
        and will be fed data during training.
    
        Adds following nodes to the computational graph
    
        input_placeholder: Input placeholder tensor of shape
                                              (batch_size, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape
                                              (batch_size, n_classes), type tf.int32
        """
        #Can set first dimensions of placeholders to "None", but I state self.batch_size for my own reference
        self.input_placeholder = tf.placeholder(dtype= tf.float32, shape = (self.batch_size, self.max_length, self.num_features))
        self.labels_placeholder = tf.placeholder(dtype= tf.float32, shape = (self.batch_size))
    
    def create_feed_dict(self, inputs_batch, labels_batch):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }
        """
        feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder : labels_batch}
        return feed_dict
    
    def add_embedding(self):
        """
        Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (self.batch_size, max_length, embed_size).
        """
        embedding = tf.nn.embedding_lookup(params = self.pretrained_embeddings, ids = self.input_placeholder )
        return embedding
    
    def find_batch_length(self, embeddings):
        """
        Each tweet has a different length. Find the length of each tweet;
        this length will be fed to dynamic_rnn so that the model knows when to stop processing
        for each tweet
        Arguments:
            embeddings: word embeddings for the tweet
        """
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
    
    def add_prediction_op(self):
        """
        Adds the RNN and LSTM ops
        """
        x = self.add_embedding() #Get the embeddings for this batch
        tweet_lengths = self.find_batch_length(x) #Get the tweet lengths
        
        init = tf.contrib.layers.xavier_initializer(uniform = True, seed= 1, dtype= tf.float32)
        cell_to_use = tf.contrib.rnn.LSTMCell(self.num_hidden_units, initializer = init, activation = 'tanh')
        cell_to_use = tf.contrib.rnn.DropoutWrapper(cell_to_use, output_keep_prob = 1 - self.num_dropout)
        
        output, state = tf.nn.dynamic_rnn(cell_to_use, inputs = x, sequence_length = tweet_lengths, dtype = tf.float32)
        final_cell_output = output[:, -1, :] #Slice just the last column of the second layer. 
        #final_cell_output should have dimensions (batch_size, num_hidden_units)
        return final_cell_output
    
    def add_loss_function(self, final_cell_output):
        """
        Adds the loss_function
        """
        init = tf.contrib.layers.xavier_initializer(uniform = True, seed= 1, dtype= tf.float32)
        class_weights = tf.get_variable("class_weights", initializer = init, shape = (self.num_hidden_units, self.num_classes))
        class_bias = tf.get_variable("class_bias", initializer = init, shape = (self.num_classes))
        predictions = tf.matmul(final_cell_output, class_weights) + class_bias #(batch_size x num_classes output)
        
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.labels_placeholder, logits = predictions)
        loss = tf.reduce_mean(loss)
        return loss
    
    def add_training_op(self, loss):
        """
        Adds the training op. Loss is the loss value given by add_loss_function
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss)
        
        return train_op
    
    def get_minibatches(self, data, x_cols, labels, b_size):
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
            print ("There are fewer samples than the batch size. Error!")
            sys.exit()
                        
        reshuffled_data = data.sample(frac=1) #Get a dataframe of completely reshuffled rows
        num_batches = math.floor(data.shape[0]/b_size) #Number of batches, if b_size is not a factor some data will be left out. Ah well.
        
        batches = []         
        for i in range(num_batches):
            start = i*b_size
            sample = reshuffled_data.iloc[start:(start + b_size)] # Sample of b size
            batches.append((sample[x_cols].values, sample[labels].values))
        
        return batches
        
        
        
        
        
        

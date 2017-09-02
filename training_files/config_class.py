# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:26:02 2017

@author: Moomootank
"""

class Config():
    def __init__(self, n_features, n_classes, batch, val_batch, test_batch, n_epochs, lr, max_l, n_layers, embeddings):
        """
        Store information about data hyperparameters. These are fixed. 
        n_features = number of columns (independent variables) in data i.e dimension of word embedding
        n_classes = number of classes to predict
        batch = batch size to use in stochastic gradient descent
        val_batch = batch size of validation set
        test_batch = batch size of test set
        
        n_epochs = number of epochs to train for 
        lr = learning rate    
        max_l = max length that RNN can extend for (i.e. longest sequence to parse)
        n_layers = number of layers in the network (i.e. how deep the network is)
        embeddings = pretrained embeddings that we will use
        """
        self.embed_size = n_features
        self.num_classes = n_classes
        self.batch_size = batch
        self.val_size = val_batch
        self.test_size = test_batch
        
        self.num_epochs =  n_epochs
        self.learning_rate = lr
        self.max_length = max_l
        self.num_layers = n_layers
        self.pretrained_embeddings = embeddings 
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:43:51 2017

@author: Moomootank
"""
import tensorflow as tf
from lstm_model import LSTM_Model

class Bidirectional_LSTM(LSTM_Model):    
    def add_cells(self):
        forward_cells = []
        backward_cells = []
        assert(len(self.num_hidden_units)==len(self.num_dropout))
        assert(len(self.num_hidden_units)==self.num_layers)
        
        init = tf.contrib.layers.xavier_initializer(uniform = True, dtype= tf.float32) 
        #For simplicity we will use the same dropout rates and hidden_unit_sizes for both the forward and backward passes
        #In reality, if we had more time and computing power, it may be better to have different dropout rates(?)
        for i in range(self.num_layers):
            without_dropout = tf.contrib.rnn.LSTMCell(self.num_hidden_units[i], activation = tf.tanh, initializer = init)
            one_layer = tf.contrib.rnn.DropoutWrapper(without_dropout, input_keep_prob = 1 - self.input_dropout_placeholder[i], 
                                                      output_keep_prob = 1 - self.dropout_placeholder[i])
            forward_cells.append(one_layer)
        
        for i in range(self.num_layers):
            #Doing this separately from the forward_cells, just to be safe (in case it's pass by reference and the model gets screwed)
            without_dropout = tf.contrib.rnn.LSTMCell(self.num_hidden_units[i], activation = tf.tanh, initializer = init)
            one_layer = tf.contrib.rnn.DropoutWrapper(without_dropout, input_keep_prob = 1 - self.input_dropout_placeholder[i], 
                                                      output_keep_prob = 1 - self.dropout_placeholder[i])
            backward_cells.append(one_layer)
        
        multi_forward_cell = tf.contrib.rnn.MultiRNNCell(forward_cells)
        multi_backward_cell = tf.contrib.rnn.MultiRNNCell(backward_cells)
        return multi_forward_cell , multi_backward_cell
    
    #=====The following functions define the main operations of the model=====
    def add_prediction_op(self):
        x = self.add_embedding()
        tweet_lengths = self.find_batch_length(x) 
        
        with tf.name_scope("Prediction_ops"):
            init = tf.contrib.layers.xavier_initializer(uniform = True, dtype= tf.float32)           
            forward_cell, backward_cell = self.add_cells()
            
            outputs, state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, inputs = x, sequence_length = tweet_lengths, dtype = tf.float32)
            
            '''
            outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output Tensor. 
            If time_major == False (default), output_fw will be a Tensor shaped: [batch_size, max_time, cell_fw.output_size] and output_bw will be a Tensor shaped: 
            [batch_size, max_time, cell_bw.output_size]
            '''
            forward_output = outputs[0][:,-1] #[batch_size, num_hidden units]
            backward_output = outputs[1][:, -1] 
            output = tf.concat([forward_output, backward_output], axis = 1) #Double check if axis =1; should be

            class_weights = tf.get_variable("class_weights", shape = (self.num_hidden_units[-1]*2, self.num_classes))
            class_bias = tf.get_variable("class_bias", initializer = init, shape = (self.num_classes))
            
            predictions = tf.matmul(output, class_weights) + class_bias #(batch_size, 2*n_hidden_units) x (2*n_hidden_units, num_classes)
        return predictions
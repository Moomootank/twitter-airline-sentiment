# -*- coding: utf-8 -*-
"""
This script converts the raw glove and tweet data into data which the neural network can use
"""
import re
import pandas as pd
import numpy as np
import pickle
import copy

from bs4 import BeautifulSoup
from nltk import TweetTokenizer
from sklearn.model_selection import train_test_split

def clean_tweets(data, tweet_col):
    '''
    Function that cleans tweets to remove noise and makes them suitable for analysis
    Arguments:
        data: Dataframe which contains the tweets to be cleaned
        tweet_col: Name of column in dataframe to contain the tweets
    '''
    df = data.copy()
    
    df[tweet_col] = df[tweet_col].map(lambda x: BeautifulSoup(x, 'lxml').get_text()) #Remove html tokens
    df[tweet_col] = df[tweet_col].map(lambda x: x.lower())
    
    #There are ways to do all the subtitutions in one go. Explore them when you have time. Doesn't seem to work well if reg exps though
    df[tweet_col] = df[tweet_col].apply(lambda x: re.sub(r"(\S*@\S*\.\S*)", " email address ",x)) #Replace emails with generic email
    df[tweet_col] = df[tweet_col].apply(lambda x: re.sub(r"(?:http\S+)|#\S+|\.|-", " ", x)) #Remove links, hashtags, periods and hyphens
    df[tweet_col] = df[tweet_col].apply(lambda x: re.sub(r"(\d+)", " digit ",x)) #Replace all numbers with "digit"
    df[tweet_col] = df[tweet_col].apply(lambda x: re.sub(r"'", "", x)) #Replace apostrophes with nothing (instead of a space)
        
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True) 
    tokens = df[tweet_col].apply(lambda x: tokenizer.tokenize(x))
    
    return df, tokens

def decrypt_glove(glove_url, dim):
    """
    This function converts the file of pretrained glove vectors into a numpy array for use in tensorflow
    Arguments:
        glove_url: Location URL of the glove repository
        dim: dimension of the word vectors
    """
    glove_file = open(glove_url, encoding = 'utf-8').read().split('\n')    
    holder = []
    index_dict = {}
    
    current = 0
    non_eng = []

    for index in range(len(glove_file)):
        line = glove_file[index].split()
        try:
            if len(line)!=(dim+1):
                #There might be some word vectors with errors in length. I know of at least one
                print ("The word embedding for {word} is not of the required dimensionality".format(word=line[0]))
                print ("Embedding is of length: ", len(line))
                print ("Index is:" , index)
                continue
           
            word = line[0]         
            word.encode('ascii') #Try to encode it to check if it is ascii. If an error is thrown, the word is not english
            index_dict[word] = current
            holder.append(line[1:]) #Exclude index 0, as that is the word itself
            current = current + 1
        except IndexError:
            print ("Null vector triggered at index:", index)
            print ("Line is: ", line)
            continue
        except UnicodeEncodeError:
            non_eng.append(word)
            continue

    #Add the word embedding for the unknown token <unk> to holder. It will be the last row in the embedding matrix
    embeddings = np.array(holder, dtype= "float32")
    print ("Initial embedding shape:", embeddings.shape)
    
    np.random.seed(1) #Set the seed to ensure that results can be reproduced
    unk_embed = np.random.normal(np.mean(embeddings,axis=0), np.var(embeddings, axis = 0)) #Generate a random word emebdding with mean and var of the others
    embeddings = np.append(embeddings, [unk_embed], axis=0) #Append the word embedding for unknown tokens to the embeddings matrix
    index_dict["<unk>"] = embeddings.shape[0] - 1 #Set unk token to last row of embeddings matrix
    print ("Embedding shape after unk:", embeddings.shape)
    
    embeddings = np.append(embeddings, [np.zeros(embeddings.shape[1])], axis = 0) 
    #Create and append the vector of zeroes. This vector will be used for padding
    index_dict["<padding>"] = embeddings.shape[0] - 1 #Now that we have added one more row, set the padding index to the last row
    print ("Embedding shape after unk and padding", embeddings.shape)
    
    return index_dict, embeddings, non_eng

def create_embeddings_numbers(tokens, index_dict):
    """
    Function that takes a list of tokens and maps it to a list of indices. Each index represents the row of that token in the glove embedding array
    Arguments:
        df: dataframe of tweets
        token_col: name of column which contains the list of tokens for each tweet
        index_dict: Dictionary of (token, row index in embedding matrix). Obtained from glove file
    """
    unknowns = []
    
    def process_list(my_list):
        """
        Helper function that converts each list of tokens into a list of indices
        Arguments:
            my_list: The list of tokens to process
        """
        holder = []
        unk_index = index_dict["<unk>"] #Row index of embedding matrix that contains vector rep of the unk token
        for item in my_list:
            if item in index_dict.keys():
                holder.append(index_dict[item])
            else:
                unknowns.append(item)
                holder.append(unk_index) #The last row of the embedding matrix will be the <unk> token
        return holder
    
    embedding_indices = tokens.apply(lambda x: process_list(x))
    return unknowns, embedding_indices

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
            copied_tweet.extend([zero_embedding_index for i in range(number_to_extend)])
            holder.append(copied_tweet)
    
    return np.array(holder, dtype= np.int32)

def save_obj(obj, directory):
    with open(directory,'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_obj(directory):
    with open(directory, 'rb') as file:
        return pickle.load(file)
        
if __name__ == "__main__":
    tweets_url = r"D:\Data Science\Projects\twitter-airline-sentiment\raw_data\Tweets.csv"
    tweets_raw = pd.DataFrame.from_csv(tweets_url)
    tweets = tweets_raw['text']
    
    glove_url = r"D:\Data Science\Projects\twitter-airline-sentiment\raw_data\glove.twitter.27B.200d.txt"
    index_dict, embedding_matrix, non_eng = decrypt_glove(glove_url, 200)
    embedding_matrix = (embedding_matrix - np.mean(embedding_matrix, axis = 0))/np.std(embedding_matrix, axis = 0)
    
    save_obj(embedding_matrix, r"D:\Data Science\Projects\twitter-airline-sentiment/training_files/training_data/embedding_matrix.pickle")

    tweets_clean, tokens = clean_tweets(tweets_raw, 'text')
    tweets_clean.to_csv(r"../training_files/training_data/tweets_clean.csv")

    unknowns, embedding_indices = create_embeddings_numbers(tokens, index_dict)
    padded_indices = pad_sequences(embedding_indices, 35, embedding_matrix.shape[0] - 1)
    
    save_obj(padded_indices, r"../training_files/training_data/padded_indices.pickle")
    
    one_hot_labels = pd.get_dummies(tweets_clean['airline_sentiment']).values
    #3d array 1 hot array; negative, neural, positive labels
    save_obj(one_hot_labels, r"../training_files/training_data/labels.pickle")
    
    #Split the data 80-10-10 into train, validation and test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(padded_indices, one_hot_labels, test_size = 0.1, stratify=one_hot_labels)
    save_obj(x_test,r"../training_files/training_data/test_data/test_indices.pickle")
    save_obj(y_test,r"../training_files/training_data/test_data/test_labels.pickle")
    
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size = 0.1, stratify=y_train_val)
    save_obj(x_train, r"../training_files/training_data/train_data/train_indices.pickle")
    save_obj(y_train, r"../training_files/training_data/train_data/train_labels.pickle")
    
    save_obj(x_val, r"../training_files/training_data/validation_data/validation_indices.pickle")
    save_obj(y_val, r"../training_files/training_data/validation_data/validation_labels.pickle")
    

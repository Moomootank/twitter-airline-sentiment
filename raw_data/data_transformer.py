# -*- coding: utf-8 -*-
"""
This script converts the raw glove and tweet data into data which the neural network can use
"""
import re
import pandas as pd
import numpy as np
import pickle

from bs4 import BeautifulSoup
from nltk import TweetTokenizer

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
    df['tokens'] = df[tweet_col].apply(lambda x: tokenizer.tokenize(x))
    
    return df

def create_embeddings_numbers(df, token_col, index_dict):
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
        maximum_value = max(index_dict.values()) #Last row of the embedding matrix
        for item in my_list:
            if item in index_dict.keys():
                holder.append(index_dict[item])
            else:
                unknowns.append(item)
                holder.append(maximum_value) #The last row of the embedding matrix will be the <unk> token
        return holder
    
    df['embed_indices'] = df[token_col].apply(lambda x: process_list(x))
    return unknowns
    
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
    
    offset = 0
    
    for index in range(len(glove_file)):
        line = glove_file[index].split()
        try:
            if len(line)!=(dim+1):
                print ("The word embedding for {word} is not of the required dimensionality".format(word=line[0]))
                print ("Embedding is of length: ", len(line))
                offset = offset + 1
                continue
            
            index_dict[line[0]] = index - offset 
            #Store the row_index of this word in a dict. If a previous line was wrong, then you have to offset the index of this line
            holder.append(line[1:]) #Exclude index 0, as that is the word itself
        except IndexError:
            print ("Null vector triggered at index: ", index)
            print ("Line is: ", line)
            offset = offset + 1
            continue
    
    print("Offset is: ", offset)
    
    #Add the word embedding for the unknown token <unk> to holder. It will be the last row in the embedding matrix
    embeddings = np.array(holder, dtype= "float32")
    unk_embed = np.random.normal(np.mean(embeddings,axis=0), np.var(embeddings, axis = 0)) #Generate a random word emebdding with mean and var of the others
    embeddings = np.append(embeddings, [unk_embed], axis=0) #Append the word embedding for unknown tokens to the embeddings matrix
    index_dict["<unk>"] = len(embeddings) #Set unk token to last row of embeddings matrix
    
    return index_dict, embeddings

def save_obj(obj, directory):
    with open(directory,'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    
if __name__ == "__main__":
    tweets_url = r"Tweets.csv"
    tweets_raw = pd.DataFrame.from_csv(tweets_url)
    tweets = tweets_raw['text']
    
    glove_url = r"glove.twitter.27B.100d.txt"
    index_dict, embedding_matrix = decrypt_glove(glove_url, 100)
    
    tweets_clean = clean_tweets(tweets_raw, 'text')
    unknowns = create_embeddings_numbers(tweets_clean, "tokens", index_dict)
    
    save_obj(embedding_matrix, r"../training_files/embedding_matrix")
    save_obj(tweets_clean, r"../training_files/tweets_clean" )
    
    
        
    
    
# -*- coding: utf-8 -*-
"""
This script converts the raw data file tweets.json into usable data

"""
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import pickle

#=====The following functions convert the raw json files into data that we can use=====
def decrypt_json(data_url):
    """
    This function converts the file tweets.json into tweets data where each row is:
    Screen_name tweet_text target_variable
    
    Arguments:
        data_url: Location URL of tweets.json
    """
    read_raw = open(data_url).read().split('\n')
    read_raw = read_raw[:-1]
    #Data is split by \n. Result is a list of json objects. Remove index -1 as it is just a None (due to data organization)
    holder = []
    
    for item in read_raw:
        read_item = json.loads(item)
        to_append = (read_item['screen_name'], read_item['text'])
        holder.append(to_append)
        
    df = pd.DataFrame(holder, columns =['name', 'text'])
    return df
        
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
    return index_dict, np.array(holder)  

#=====The following functions clean the tweet data to remove noise =====

def clean_tweets(df, tweet_col):
    """
    This function removes urls from all the tweets. All urls will begin with http
    Note: Links shared on Twitter, including links shared in Direct Messages,
    will automatically be processed and shortened to an http://t.co link. 
    
    It will also remove: apostraphes, html information, 
    
    Arguments:
        df: Dataframe to remove urls from
        tweet_col: Name of column that contains the tweet data
    """
    
    df[tweet_col] = df[tweet_col].map(lambda x: x.lower()) #Should this be the first?...
    df[tweet_col] = df[tweet_col].map(lambda x: BeautifulSoup(x, 'lxml').get_text()) #removes html parsers
    df[tweet_col] = df[tweet_col].map(lambda x: re.sub(r"(?:\@|http)\S+|'", "", x)) #Replaces anything starting with @ or http with "", or is an apostraphe  
    df[tweet_col] = df[tweet_col].map(lambda x: re.sub(r"/"," or ", x)) #Converts / to or eg. low/mid to low or mid


#=====The following functions save and load objects      
def save_obj(obj, name):
    with open('{name}.pickle'.format(name=name), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('{name}.pickle'.format(name=name), 'rb') as handle:
        return pickle.load(handle)
            
if __name__ == "__main__":
    data_url = r"D:\Data Science\Projects\US_PoliticalTweets\raw_data\tweets.json"
    #tweets_data = decrypt_json(data_url)
    #tweets_data.to_csv("tweets_data.csv")
    tweets = pd.read_csv(open('tweets_data.csv','r'), encoding='utf-8', engine='c', index_col = 0) #Have to code it this way to avoid a buffer overflow
   
    glove_url = r"D:\Data Science\Projects\US_PoliticalTweets\raw_data\glove.twitter.27B.100d.txt"
    #index_dict, embeddings = decrypt_glove(glove_url, 100)
    #save_obj(embeddings, "glove_100d")
    #save_obj(index_dict, "glove_100d_index_dict")
    embeddings = load_obj("glove_100d")
    index_dict = load_obj("glove_100d_index_dict")
    

    
    
    
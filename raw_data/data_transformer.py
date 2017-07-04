# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def clean_tweets(df, tweet_col):
    '''
    Function that cleans tweets to remove noise and makes them suitable for analysis
    Arguments:
        df: Dataframe which contains the tweets to be cleaned
        tweet_col: Name of column in dataframe to contain the tweets
    '''
    
    df[tweet_col] = df[tweet_col].map(lambda x: BeautifulSoup(x, 'lxml').get_text())
    df[tweet_col] = df[tweet_col].map(lambda x: x.lower())
    
    #There are ways to do all the subtitutions in one go. Explore them when you have time
    df[tweet_col] = df[tweet_col].apply(lambda x: re.sub(r"(?:http\S+)", "<link>", x)) #Doesn't work with match_dict for some reason
    df[tweet_col] = df[tweet_col].apply(lambda x: 
        re.sub(r"(@virginamerica|@jetblue|@united|@southwestair|@jetblue|@usairways|@americanair)", "@ airline", x ))
    df[tweet_col] = df[tweet_col].apply(lambda x: re.sub(r"@\S+", "@ <user>", x))
        

    

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

if __name__ == "__main__":
    tweets_url = r"D:\Data Science\Projects\twitter-airline-sentiment\Tweets.csv"
    tweets_raw = pd.DataFrame.from_csv(tweets_url)
    
    glove_url = r"D:\Data Science\Projects\twitter-airline-sentiment\raw_data\glove.twitter.27B.100d.txt"
    glove_data = decrypt_glove(glove_url, 100)
    
    
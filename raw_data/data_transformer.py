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
    df[tweet_col] = df[tweet_col].apply(lambda x: re.sub(r"(?:http\S+)", "<link>", x))
    df[tweet_col] = df[tweet_col].apply(lambda x: 
        re.sub(r"(@virginamerica|@jetblue|@united|@southwestair|@jetblue|@usairways|@americanair)", )

    df[tweet_col] = df[tweet_col].map(lambda x: x.lower())
if __name__ == "__main__":
    tweets_url = r"D:\Data Science\Projects\twitter-airline-sentiment\Tweets.csv"
    tweets_raw = pd.DataFrame.from_csv(tweets_url)
    
    
    
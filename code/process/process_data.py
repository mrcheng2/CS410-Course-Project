import pandas as pd
import numpy as np


class DataProcessor:

    """
    Data class to fetch raw data and process it
    """

    def __init__(self):
        self.fetch_raw_data()

    def fetch_raw_data(self):
        """
        Fetch raw data
        """
        biden_tweets = pd.read_csv('../../data/train/raw/hashtag_joebiden.csv', encoding_errors='ignore', lineterminator='\n')
        trump_tweets = pd.read_csv('../../data/train/raw/hashtag_donaldtrump.csv', encoding_errors='ignore', lineterminator='\n')
        all_tweets = pd.concat([biden_tweets, trump_tweets])
        all_tweets = all_tweets.drop_duplicates('tweet_id')
        all_tweets['timestamp'] = pd.to_datetime(all_tweets['created_at'])
        self.df_data = all_tweets

    def process_data(self,
                     only_US: bool,
                     only_english: bool):
        """
        Filter and process raw data
        Args:
            only_US: True to only filter data coming from US
            only_english: True to only filter data where tweets are in English
        
        Returns:
            df_processed: Dataframe with 2 columns: i) timestamp ii) textdata
        """
        
        df_processed = self.df_data.copy()
        df_processed = df_processed.rename(columns = {'tweet': 'textdata'})
        if only_US:
            df_processed = df_processed.loc[df_processed['country'] == 'United States of America']
        
        if only_english:
            pass
        
        df_processed = df_processed[['timestamp', 'textdata']]
        return df_processed
        

        


import pandas as pd
import numpy as np


class NewsDataProcessor:

    """
    Data class to fetch raw data and process it
    """

    def __init__(self):
        self.fetch_raw_data()

    def fetch_raw_data(self):
        """
        Fetch raw data
        """
        articles = pd.read_json('data/articles_Oct5-to-Nov4.json')
        # print(articles.head())
        self.df_data = articles

    def process_data(self):
        """
        Filter and process raw data
        Args:
            only_US: True to only filter data coming from US
            only_english: True to only filter data where tweets are in English
        
        Returns:
            df_processed: Dataframe with 2 columns: i) timestamp ii) textdata
        """
        
        df_processed = self.df_data.copy()
        df_processed = df_processed.rename(columns = {'title': 'textdata'}) #rename title to textdata
        df_processed = df_processed.rename(columns = {'publishedAt': 'timestamp'}) #rename publishedAt to timestamp
        
        columns_to_keep = ['textdata', 'timestamp']
        df_processed = df_processed.loc[:, columns_to_keep] #keep just textdata and timestamp columns
        df_processed = df_processed[df_processed['textdata'] != '[Removed]'] #remove rows where there is no title
        
        mask = df_processed['textdata'].str.contains('trump|harris|walz|vance|biden|election|president|presidential|campaign', case=False, na=False)

        df_processed = df_processed[mask]

        json_string = df_processed.to_json(orient='records')
        json_string = f"[{json_string}]"

        with open('data/processed_news.json', 'w') as json_file:
            json_file.write(json_string)
        
        # print(df_processed)

        return df_processed
        

processor = NewsDataProcessor()
processor.process_data()


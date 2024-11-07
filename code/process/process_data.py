import pandas as pd
import os
import re
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

class DataProcessor:

    """
    Data class to fetch raw data and process it.

    Sample call:

    #Dump the processed file into  the processed data folder
    data_processor = DataProcessor()
    processed_data = data_processor.process_data(dump_processed = True)
    """

    def __init__(self):
        self.fetch_raw_data()

    def fetch_raw_data(self):
        """
        Fetch raw data
        """
        biden_tweets = pd.read_csv('../../data/train/raw/hashtag_joebiden.csv', encoding_errors='ignore', lineterminator='\n')
        biden_tweets['candidate'] = 'other'
        trump_tweets = pd.read_csv('../../data/train/raw/hashtag_donaldtrump.csv', encoding_errors='ignore', lineterminator='\n')
        trump_tweets['candidate'] = 'trump'
        all_tweets = pd.concat([biden_tweets, trump_tweets])
        all_tweets = all_tweets.drop_duplicates('tweet_id')
        all_tweets['timestamp'] = pd.to_datetime(all_tweets['created_at'])
        
        self.df_data = all_tweets

    def process_data(self,
                     dump_processed: bool,
                     processed_filename: str = 'processed_data'):
        """
        Filter and process raw data

        dump_processed: True to dump processed file
        processed_filename: Name of processed file
        
        Returns:
            df_processed: Dataframe with 2 columns: i) timestamp ii) textdata
        """
        
        #Load and label the data
        nlp = spacy.load('en_core_web_sm')  # 1
        def create_lang_detector(nlp, name):
            return LanguageDetector()
        Language.factory("language_detector", func=create_lang_detector)
        nlp.add_pipe('language_detector', last=True)
        self.df_data['is_en'] = self.df_data['tweet'].apply(lambda x: self.check_en(x, nlp))
        
        df_processed = self.df_data.copy()
        df_processed = df_processed.rename(columns = {'tweet': 'textdata'})
        df_processed['textdata'] = df_processed['textdata'].apply(self.clean_text)
        
        if dump_processed:
            #Check if file dir exists
            filedir = '../../data/train/processed/'
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            filename = os.path.join(filedir, f'{processed_filename}.csv')
            #dump file
            df_processed.to_csv(filename)
            print(f'File dump into: {filename}')
        return df_processed

    @staticmethod
    def clean_text(text):
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
        text = re.sub(r"[^a-zA-Z0-9\s]", '', text)  # Remove special characters
        return text.lower().strip()  # Convert to lowercase and strip extra spaces

    @staticmethod
    def check_en(txt_data, nlp_obj):
        doc = nlp_obj(txt_data) #3
        detect_language = doc._.language #4
        return detect_language.get('language', None) == 'en'




        

        


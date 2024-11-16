import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tqdm import tqdm
import json
from pandas import json_normalize

tqdm.pandas()

# Download VADER lexicon
nltk.download('vader_lexicon')

class SentimentAnalysis:

    COLS_KEEP = ['textdata', 'timestamp','sentiment_score', 'sentiment_label']
    
    def __init__(self, input_file_path, output_file_path, threshold=0.05):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.threshold = threshold
        self.sid = SentimentIntensityAnalyzer()
        f = open(input_file_path)
        data =json.load(f)
        for a in data:
            self.df=pd.DataFrame(a,columns=['textdata', 'timestamp'])
    
    def process_headlines(self):

        print("Calculating polarity scores and labeling sentiments for each tweet...")
        print(self.df)
        self.df['sentiment_score'] = self.df['textdata'].progress_apply(
            lambda text: self.sid.polarity_scores(text)['compound']
        )
        self.df['sentiment_label'] = self.df['sentiment_score'].progress_apply(self.label_sentiment)
        self.df = self.df[self.COLS_KEEP]
        self.df.to_csv(self.output_file_path, index=False)

    def label_sentiment(self, score):
        """Labels sentiment based on the score and threshold."""
        if score >= self.threshold:
            return "Positive"
        elif score <= -self.threshold:
            return "Negative"
        else:
            return "Neutral"


input_file = '../../data/processed_news.json'
output_file = '../../data/test/processed/VADER_processed_data_news.csv'
analyzer = SentimentAnalysis(input_file, output_file)
analyzer.process_headlines()

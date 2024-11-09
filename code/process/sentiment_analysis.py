import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tqdm import tqdm

tqdm.pandas()

# Download VADER lexicon
nltk.download('vader_lexicon')

class SentimentAnalysis:

    COLS_KEEP = ['timestamp', 'country', 'candidate', 'is_en', 'sentiment_score', 'sentiment_label', 'likes', 'user_join_date', 'user_followers_count']
    
    def __init__(self, input_file_path, output_file_path, threshold=0.05):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.threshold = threshold
        self.df = pd.read_csv(self.input_file_path)
        self.sid = SentimentIntensityAnalyzer()

    def process_tweets(self):
        # Filter English tweets
        self.df = self.df[self.df['is_en'] == True].copy()

        print("Calculating polarity scores and labeling sentiments for each tweet...")
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


input_file = '../../data/train/processed/processed_data.csv'
output_file = '../../data/train/processed/VADER_processed_data.csv'
analyzer = SentimentAnalysis(input_file, output_file)
analyzer.process_tweets()

df = pd.read_csv(output_file)

biden_tweets = df[df['candidate'] == 'other']
trump_tweets = df[df['candidate'] == 'trump']

biden_sentiment_avg = biden_tweets['sentiment_score'].mean()
biden_sentiment_counts = biden_tweets['sentiment_label'].value_counts()

print("Biden - Average Sentiment Score:", biden_sentiment_avg)
print("Biden - Sentiment Counts:\n", biden_sentiment_counts)
print("===================")

trump_sentiment_avg = trump_tweets['sentiment_score'].mean()
trump_sentiment_counts = trump_tweets['sentiment_label'].value_counts()

print("Trump - Average Sentiment Score:", trump_sentiment_avg)
print("Trump - Sentiment Counts:\n", trump_sentiment_counts)

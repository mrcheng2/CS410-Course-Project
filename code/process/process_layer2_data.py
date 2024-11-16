import pandas as pd
from process_label_data import LabelDataProcessor
import numpy as np

class SecondLayerDataHandler:

    """
    Base second layer datahandler to format processed data
    Also:
    i) Chooses which other features to add
    ii) Chooses which tweets/newsheadlines to filter out

    This class is the base class to be used by subclasses for other data manipulation
    """

    DATANAME = '2RATIOS'
    def __init__(self, sentiment_model: str):
        self.sentiment_model = sentiment_model
        self.df_sentiment_data = self.get_sentiment_data()
        self.df_label_data = LabelDataProcessor.get_label_data()
        self.df_all_data = self.format_predictor()

    def get_sentiment_data(self):
        df_sentiment_data = pd.read_csv(f'../../data/train/processed/{self.sentiment_model}_processed_data.csv')
        df_sentiment_data.index = pd.to_datetime(df_sentiment_data['timestamp'])
        df_sentiment_data['date'] = df_sentiment_data.index.date
        return df_sentiment_data

    def format_predictor(self):
        positive_label = {'Neutral' : 0, 'Positive': 1, 'Negative': -1}
        df_data = self.df_sentiment_data.copy()
        df_data['istrump'] = np.where(df_data['candidate'] == 'trump', 1, 0)
        df_data['sentiment_indic'] = df_data['sentiment_label'].map(positive_label)
        df_grouped_data = df_data.groupby(['date', 'candidate']).aggregate({'sentiment_indic' : ['sum', 'count']}).reset_index().set_index(['date', 'candidate'])
        df_grouped_data.columns = df_grouped_data.columns.get_level_values(1)
        df_grouped_data['ratio'] = df_grouped_data['sum']/df_grouped_data['count']
        df_predict_data = df_grouped_data.reset_index().pivot(index = 'date', columns = 'candidate', values = 'ratio')
        df_all_data = pd.merge(df_predict_data, self.df_label_data[['p_trump_win']], left_index = True, right_index = True)
        return df_all_data

    def __repr__(self):
        return f'SecondLayerDataHandler:{self.DATANAME}'

class OneRatioSecondLayerDataHandler(SecondLayerDataHandler):
    DATANAME = '1RATIO'

    def format_predictor(self):
        positive_label = {'Neutral' : 0, 'Positive': 1, 'Negative': -1}
        df_data = self.df_sentiment_data.copy()
        df_data = df_data.loc[df_data['candidate'] == 'trump']
        df_data['sentiment_indic'] = df_data['sentiment_label'].map(positive_label)
        df_grouped_data = df_data.groupby(['date']).aggregate({'sentiment_indic' : ['sum', 'count']}).reset_index().set_index(['date'])
        df_grouped_data.columns = df_grouped_data.columns.get_level_values(1)
        df_grouped_data['ratio'] = df_grouped_data['sum']/df_grouped_data['count']
        df_predict_data = df_grouped_data[['ratio']]
        df_all_data = pd.merge(df_predict_data, self.df_label_data[['p_trump_win']], left_index = True, right_index = True)
        return df_all_data


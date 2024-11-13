import pandas as pd
import numpy as np


class TestLabelDataProcessor:

    """
    Load the test label data. Data found in daily_summary.csv file from https://projects.fivethirtyeight.com/2024-election-forecast/.
    """

    def __init__(self):
        self.df_label_data = self.get_label_data()

    def get_label_data(self):
        """
        Fetch raw data
        """
        label_data = pd.read_csv('../../data/test/raw/daily_summary.csv')
        label_data = label_data[(label_data['state_abb'].isna()) & (label_data['variable'] == 'electoral college') & (label_data['metric'] == 'p_win')]
        label_data = label_data.pivot(index='model_date', columns='party', values='value')
        label_data = label_data.rename(columns = {'REP': 'p_trump_win', 'DEM': 'p_trump_lose_1', 'IND': 'p_trump_lose_2'})
        label_data.fillna(0, inplace=True)
        label_data['p_trump_lose'] = label_data['p_trump_lose_1'] + label_data['p_trump_lose_2']
        return label_data[['p_trump_win', 'p_trump_lose']]

        


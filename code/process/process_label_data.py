import pandas as pd
import os
import re

class LabelDataProcessor:

    """
    Load the train label data
    """

    def __init__(self):

        self.df_label_data = self.get_label_data()

    @staticmethod
    def get_label_data():
        label_data = pd.read_csv('../../data/train/raw/presidential_national_toplines_2020.csv')
        label_data.index = pd.to_datetime(label_data['modeldate'])
        label_data = label_data.rename(columns = {'ecwin_inc': 'p_trump_win', 'ecwin_chal': 'p_trump_lose'})
        return label_data[['p_trump_win', 'p_trump_lose']]


import pandas as pd
import os
import sys
import re

sys.path.append('../process/')
from process_layer2_data import SecondLayerDataHandler, OneRatioSecondLayerDataHandler
import numpy as np

import statsmodels.api as sm

class ProbabilityModeler:

    """
    Probability modeler class to predict Trump probability.
    This is the baseclass used for probability modelling. 

    Subclasses can change up the run_model method to implement its own model
    """

    MODELNAME = "OLS"
    
    def __init__(self,
                 sentiment_model: str,
                 layer2_datahandler: SecondLayerDataHandler,
                 verbose: bool = True):
        self.sentiment_model = sentiment_model
        self.layer2_datahandler = layer2_datahandler(sentiment_model = self.sentiment_model)
        self.verbose = verbose

    def __repr__(self):
        return f'ProbabilityModeler:{self.MODELNAME}'

    @property
    def params_outputfile(self):
        return f'../params/{self.sentiment_model}_{self.layer2_datahandler.DATANAME}_{self.MODELNAME}_params.csv'

    def run_model(self, y_var: str = 'p_trump_win'):
        y_var = 'p_trump_win'
        df_data = self.layer2_datahandler.df_all_data
        x_vars = [col for col in df_data.columns if col != y_var]
        model = sm.OLS(df_data[y_var], df_data[x_vars]).fit()
        if self.verbose:
            print(f'Modelname: {self.MODELNAME},Sentiment Model: {self.sentiment_model},  rsquared: {model.rsquared:.2f}')
        self.save_model(model)
        return model

    def save_model(self, model):
        params_df = pd.DataFrame(model.params).T
        params_df.index = ['betas']
        params_df.to_csv(self.params_outputfile)
        print(f'File saved in: {self.params_outputfile}')


if __name__ == "__main__":

    prob_modeler = ProbabilityModeler(sentiment_model= 'VADER', layer2_datahandler = SecondLayerDataHandler)
    one_var_prob_modeler = ProbabilityModeler(sentiment_model= 'VADER', layer2_datahandler = OneRatioSecondLayerDataHandler)

    model_pipeline = [prob_modeler, one_var_prob_modeler]

    for modeler in model_pipeline:
        model = modeler.run_model()
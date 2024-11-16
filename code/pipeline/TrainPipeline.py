import pandas as pd
import os
import sys
import re

sys.path.append('../process/')
from process_layer2_data import SecondLayerDataHandler, OneRatioSecondLayerDataHandler
from sentiment_analysis import SentimentAnalysis

sys.path.append('../model/')
from model import ProbabilityModeler

import numpy as np


SENTIMENT_MODEL_MAP = {'VADER': SentimentAnalysis}

LAYER2_DATAHANDLER_MAP = {'2RATIOS': SecondLayerDataHandler,
                          '1RATIO' : OneRatioSecondLayerDataHandler}

PROBABILITY_MODEL_MAP = {'OLS': ProbabilityModeler}

class TrainModelPipeline:


    INPUT_PROCESSED_DATAFILE = '../../data/train/processed/processed_data.csv'

    def __init__(self,
                 sentiment_model_name: str,
                 layer2_process_name: str,
                 probability_model_name: str,
                 sentiment_threshold: float = 0.05,
                 run_sentiment_model: bool = True):
        
        self.sentiment_model_name = sentiment_model_name
        self.layer2_process_name = layer2_process_name
        self.probability_model_name = probability_model_name
        self.sentiment_threshold = sentiment_threshold
        self.run_sentiment_model = run_sentiment_model

        #Fetch the models
        self.fetch_models()
        self.apply_pipeline()

    def fetch_models(self):
        self.sentiment_model = SENTIMENT_MODEL_MAP.get(self.sentiment_model_name, None)
        self.layer2_processor = LAYER2_DATAHANDLER_MAP.get(self.layer2_process_name, None)
        self.probability_model = PROBABILITY_MODEL_MAP.get(self.probability_model_name, None)

        if self.sentiment_model is None:
            raise(Exception(f'Sentiment Model:{self.sentiment_model_name} is not recognized'))
        if self.layer2_processor is None:
            raise(Exception(f'Layer2 DataHandler:{self.layer2_process_name} is not recognized'))
        if self.probability_model is None:
            raise(Exception(f'Probability Model:{self.probability_model_name} is not recognized'))

    def apply_pipeline(self):
        #2) Apply the sentiment model
        if self.run_sentiment_model:
            output_sentiment_file = f'../../data/train/processed/{self.sentiment_model_name}_processed_data.csv'
            self.sentiment_analyzer = SentimentAnalysis(input_file_path = self.INPUT_PROCESSED_DATAFILE,
                                                   output_file_path = output_sentiment_file,
                                                   threshold = self.sentiment_threshold)
            self.sentiment_analyzer.process_tweets()

        #3/4) Apply Layer2DataHandler and apply probability model
        self.prob_modeler = self.probability_model(sentiment_model= self.sentiment_model_name,
                                              layer2_datahandler = self.layer2_processor)
        self.prob_modeler.run_model()
        

if __name__ == "__main__":
    #Train Model e.g. 1
    train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'VADER',
                                          layer2_process_name = '2RATIOS',
                                          probability_model_name = 'OLS',
                                          run_sentiment_model = True)

    #Train Model e.g. 2 (Do not run sentiment model again)
    train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'VADER',
                                          layer2_process_name = '1RATIO',
                                          probability_model_name = 'OLS',
                                          run_sentiment_model = False)


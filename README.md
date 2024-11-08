# CS410 Course Project

## How to Run the Project

1. **Clone the Repository**: Pull this project to your local machine using Git.

2. **Download Dataset**: Retrieve the *US Election 2020 Tweets* dataset from [Kaggle](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets/data).

3. **Organize Files**: Place `hashtag_donaldtrump.csv` and `hashtag_joebiden.csv` in the `../../data/train/raw` directory.

4. **Set Up Environment**:
   - **Mac Users**: In the directory where `environment.yml` is located, create the environment using:
     ```bash
     conda env create -f environment.yml
     ```
   - **Windows Users**: Manually create a Conda environment and install the required packages `pandas` and `spacy`.

5. **Initialize Data Processor**: Run `process_data.py` to initialize the `DataProcessor` class.

6. **Data Cleaning and Labeling**: Execute `script.py` to clean and label the data using the `DataProcessor` class. 
   - *Note*: This step may take approximately 5 hours.

7. **Sentiment Analysis**: Run `sentiment_analysis.py` to perform sentiment analysis using VADER, which will add two new columns, `sentiment_score` and `sentiment_label`, to the dataset. After completion, a file named `VADER_processed_data.csv` should appear in the `CS410-Course-Project\data\train\processed` folder.

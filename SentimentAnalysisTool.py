# SentimentAnalysisTool.py
# Author: Ryan Massie
# Date: 4/13/24
# Notes: Implements a Sentiment Analysis Tool to determine if a string has a positive, negative, or neutral sentiment.
# Steps Required Include:
# 1. Preprocess Data
# 2. Create Test/Training Set
# 3. Run Model(s)
# 4. Analyze Results

# Import Libraries
import pandas as pd
import numpy as np
import os
import nltk
import nltk.classify
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from sklearn.metrics import classification_report
# Download ntlk Libraries
nltk.download('vader_lexicon')
# Import Functions
import Preprocessing as pre

# Write dataframe to csvFile, meant to be used for debugging
#    Args:
#    - outputPath: Path to the directory where the output CSV file will be saved.
#    - outputFile: Name of the output CSV file.
#    - df: pandas DataFrame to be written to the CSV file.
def writeToCSV(outputPath, outputFile, df):
    # Create Directory if it does not exist an write data to output file
    if not os.path.exists(outputPath):
    # Create the folder and any missing parent directories
        try:
            os.makedirs(outputPath)
            print(f"Folder created: {outputPath}")
        except Exception as e:
            print(f"Error creating folder: {e}")
    df.to_csv(outputPath + outputFile, encoding='utf-8', index=False)

# Preproccesses Data and Writes Result
#   Args:
#    - inputPath: Path to the input CSV file.
#   Returns:
#    - df: dataframe with processed data
def PreprocessData(inputPath):
    df = pd.read_csv(inputPath)
    df = pre.CleanDataset(df)
    df = pre.TokenizeDataset(df)
    df = pre.RemoveStopwordDatset(df)
    df = pre.Lemmatization(df)
    #writeToCSV("preprocessedData/", "out.csv", df)
    return df

# Create Training & Test Dataset
#   Args:
#    - dataset: pandas DataFrame containing the dataset to be split.
#    - split: float value between 0 and 1 indicating the percentage of entries to include in the training dataset.
#   Return:
#    - training: pandas DataFrame to store the training dataset.
#    - test: pandas DataFrame to store the test dataset.
def createDataset(dataset, split):
    # Calculate the number of entries for training dataset based on the split percentage
    train_size = int(len(dataset) * split)

    # Randomly select indices for the training dataset
    train_indices = np.random.choice(dataset.index, train_size, replace=False)

    # Select remaining indices for the test dataset
    test_indices = dataset.index.difference(train_indices)

    # Populate training and test datasets based on selected indices
    training = dataset.loc[train_indices]
    test = dataset.loc[test_indices]

    # Reset indices for training and test datasets
    training.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    #writeToCSV("preprocessedData/", "training.csv", training)
    #writeToCSV("preprocessedData/", "test.csv", test)
    return training, test

# Implement NLP Models
# Implements nltk Vader Model for Sentiment Analysis
#   Args:
#    - testModel: pandas DataFrame containing the text data to be analyzed.
#   Returns:
#    - testModel: pandas DataFrame with the sentiment analysis results added as a new column.
def NltkVaderTestModel(testModel):
    # Create an Analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Function used to analyze individual lines
    def getSentiment(text):
        sentimentScore = analyzer.polarity_scores(text)
        if (sentimentScore['compound'] < -1/3):
            return -1
        elif (sentimentScore['compound'] > 1/3):
            return 1
        else:
            return 0
    
    # Create new column in dataframe for test results
    testModel['TestVaderSentiment'] = testModel['Sentence'].apply(getSentiment)
    return testModel

data = PreprocessData("./datasets/Emotions.csv")
results = NltkVaderTestModel(data)
print(classification_report(results['Sentiment'], results['TestVaderSentiment']))
writeToCSV("preprocessedData/", "test.csv", results)
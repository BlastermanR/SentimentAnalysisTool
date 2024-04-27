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
import tensorflow as tf
import os
import random

import nltk
import nltk.classify
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint


from sklearn.model_selection import train_test_split
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

# Implement Random
def RandomModel(testModel):
    def getRandom(text):
        return random.randint(-1, 1)
    testModel['TestRandomSentiment'] = testModel['Sentence'].apply(getRandom)

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
        if (sentimentScore['compound'] < -0.05):
            return -1
        elif (sentimentScore['compound'] > 0.05):
            return 1
        else:
            return 0
    
    # Create new column in dataframe for test results
    testModel['TestVaderSentiment'] = testModel['Sentence'].apply(getSentiment)
    return testModel

def MachineLearningModel(model):
    # Get frequency count
    freq = pd.Series(' '.join(model['Sentence']).split()).value_counts()

    # Remove rare words (x < 5)
    removeWords = freq[freq < 5].index
    model['Sentence'] = model['Sentence'].apply(lambda x: ' '.join([word for word in x.split() if word not in removeWords]))
    # ensure not empty sentences
    model = model[model['Sentence'] != '']

    # Get dataframe size
    size = len(model)
    print("Model dataframe size: ", size)

    # Convert data to list
    sentenceList   = model['Sentence'].tolist()

    # Keras Tokenizer
    maxWords = 5000
    maxLen = 200
    tokenizer = Tokenizer(num_words=maxWords)
    tokenizer.fit_on_texts(sentenceList)
    sequences = tokenizer.texts_to_sequences(sentenceList)
    X_List = pad_sequences(sequences, maxlen=maxLen)


    # Models
    labels = np.array(model['Sentiment'])
    labels = tf.keras.utils.to_categorical(labels, 3, dtype="float32")
    print("Labels #: ", len(labels))

    X_train, X_test, y_train, y_test = train_test_split(X_List, labels, random_state=0)

    # Model
    deepModel = Sequential()
    deepModel.add(layers.Embedding(maxWords, 40, input_length=maxLen))
    deepModel.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6, return_sequences=True)))
    deepModel.add(layers.Bidirectional(layers.LSTM(20, dropout=0.6)))
    deepModel.add(layers.Dense(16, activation='relu'))
    deepModel.add(layers.Dense(3,activation='softmax'))
    deepModel.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

    #Implementing model checkpoins to save the best metric and do not lose it on training
    checkpoint1 = ModelCheckpoint("SA.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1, save_weights_only=False)
    history = deepModel.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), callbacks=[checkpoint1])
    prob = deepModel.predict(X_List)
    predictedLabels = np.argmax(prob, axis=1)

    indexToSentiment = {
        0: 0,  # Neutral sentiment
        1: 1,  # Positive sentiment
        2: -1  # Negative sentiment
    }

    # Map predicted indices to sentiment labels
    predictedSentiments = [indexToSentiment[index] for index in predictedLabels]
    model['TestKerasSentiment'] = predictedSentiments
    return model


## MAIN PROGRAM LINE ##
# Path to dataset (csv file)
path = "./datasets/Tweets1.csv"

# Perform Preprocessing
data = PreprocessData(path)
results = data

# Test dataset
results = RandomModel(data)
results = NltkVaderTestModel(data)
results = MachineLearningModel(data)

# Write to output file
writeToCSV("preprocessedData/", "output.csv", results)

# Display results
print("RANDOM:")
print(classification_report(results['Sentiment'], results['TestRandomSentiment']))
print("VADER:")
print(classification_report(results['Sentiment'], results['TestVaderSentiment']))
print("KERAS BIDIRECTIONAL:")
print(classification_report(results['Sentiment'], results['TestKerasSentiment']))
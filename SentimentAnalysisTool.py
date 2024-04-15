# SentimentAnalysisTool.py
# Author: Ryan Massie
# Date: 4/13/24
# Notes:

# Import Libraries
import pandas as pd
import os
# Import Functions
import Preprocessing as pre

# Preproccesses Data and Writes Result
def PreprocessData(InputPath, OutputFile):
    OutputPath = "preprocessedData/"
    df = pd.read_csv(InputPath)
    df = pre.CleanDataset(df)
    df = pre.TokenizeDataset(df)
    df = pre.RemoveStopwordDatset(df)
    #df = pre.Lemmatization(df)
    # Create preprocessedData Directory if it does not exits
    # Write data to output file
    if not os.path.exists(OutputPath):
    # Create the folder and any missing parent directories
        try:
            os.makedirs(OutputPath)
            print(f"Folder created: {OutputPath}")
        except Exception as e:
            print(f"Error creating folder: {e}")
    df.to_csv(OutputPath + OutputFile, encoding='utf-8', index=False)

PreprocessData("./datasets/Gold.csv","out.csv")
    

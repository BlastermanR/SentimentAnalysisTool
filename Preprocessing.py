# Preprocessing.py
# Author: Ryan Massie
# Date: 4/13/24
# Notes:
# Implments functions for preprocessing data for Sentiment Analysis
# Refrence:
# https://medium.com/@maleeshadesilva21/preprocessing-steps-for-natural-language-processing-nlp-a-beginners-guide-d6d9bf7689c9
#
# Preprocesses datasets specified in 'datafiles.json' steps by perfoming the following steps:
#   1. Cleaning
#   2. Tokenization
#   3. Stopword Removal
#   4. Stemming/Lemmatization 
import pandas as pd
import sys
import re
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Cleaning requires:
#   1. lowercase
#   2. URLs
#   3. @s
#   3. punctuation
#   4. numbers
# Define a function to remove URLs from text
def CleanDataset(df):
    # Lowercase
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

    # URLs
    url_pattern = re.compile(r'https?://\S+')
    def remove_urls(text):
        return url_pattern.sub('', text)
    df['Sentence'] = df['Sentence'].apply(remove_urls)

    # @stext
    def remove_words_with_at(text):
        return ' '.join(word for word in text.split() if '@' not in word)
    df['Sentence'] = df['Sentence'].apply(remove_words_with_at)

    # Punctuation/non-text characters 
    df = df.replace(to_replace=r'[^\w\s]', value='', regex=True)

    # Digits
    df = df.replace(to_replace=r'\d', value='', regex=True)

    # Return updated dataframe
    return df

# Tokenization from nltk
def TokenizeDataset(df):
    df['Sentence'] = df['Sentence'].apply(word_tokenize)
    return df

# Stopword Removal from nltk
def RemoveStopwordDatset(df):
    stop_words = set(stopwords.words('english'))
    df['Sentence'] = df['Sentence'].apply(lambda x: [word for word in x if word not in stop_words])
    return df

# Lemmatization
# initialize lemmatizer
lemmatizer = WordNetLemmatizer()
def Lemmatization(df):
    def lemmatizeTokens(tokens):
        # convert POS tag to WordNet format
        def get_wordnet_pos(word):
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        # lemmatize tokens
        lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
        # return lemmatized tokens as a list
        return lemmas
    df['Sentence'] = df['Sentence'].apply(lemmatizeTokens)
    return df
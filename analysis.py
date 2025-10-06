"""
Title: analysis.py
Author: @dsherbini
Date: Oct 6, 2025

Analysis of data work standards.
"""

# basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# nlp packages
import spacy
import nltk
from textblob import TextBlob
from nltk.corpus import wordnet
import re # regular expressions
from nltk.corpus import stopwords # stop words
from nltk.tokenize import word_tokenize # for word tokenization
from nltk.stem import WordNetLemmatizer # for stemming words
from sklearn.feature_extraction.text import CountVectorizer # for word counts
from wordcloud import WordCloud # for creating word cloud
from nltk import ngrams # for extracting phrases
from nltk.sentiment import SentimentIntensityAnalyzer # sentiment analysis
from sentence_transformers import SentenceTransformer # for semantic embeddings

# Load standards data
data = pd.read_csv('/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/standards_with_text.csv')
print(data.head())
print(data.columns)

############################## TEXT PRE-PROCESSING ##############################

def process_text(raw_text):
    '''
    Cleans text and prepares for analysis by:
        - removing punctuation
        - making lowercase
        - tokenizing text
        - removing stop words
        - stemming: breaking down words to their root
    
    Parameters
    ----------
    raw_text: string of original raw text

    Returns
    -------
    final_text: string of new processed text
    '''
    
    # Remove punctiation
    pattern = re.compile(r'[^\w\s]')
    clean_text1 = re.sub(pattern, '', raw_text).strip()
    
    # Make lowercase
    clean_text2 = clean_text1.lower().strip()
    
    # Tokenize
    clean_text3 = word_tokenize(clean_text2) 
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    clean_text4 = [w for w in clean_text3 if w not in stop_words]
    
    # Stem/lemmatize words
    lemmatizer = WordNetLemmatizer()
    final_text = [lemmatizer.lemmatize(w) for w in clean_text4]
    
    return final_text

def process_standards(data):
    '''
    Processes raw_text column in the dataframe.

    Parameters
    ----------
    data: dataframe containing standards/policy documents as raw text

    Returns
    -------
    data_clean: updated dataframe with new column for clean raw text
    '''
    text_to_clean = list(data['raw_text'])
    text_clean = [process_text(r) for r in text_to_clean]
    data_clean = data.copy()
    data_clean['clean_text'] = text_clean
    return data_clean

# Get processed text
data_clean = process_standards(data)

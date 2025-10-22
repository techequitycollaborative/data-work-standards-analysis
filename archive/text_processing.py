"""
Title: text_processing.py
Author: @dsherbini
Date: Oct 6, 2025

In this script, we generate embeddings for full policy documents and for framework parameters, using a pre-trained model.
We then compute cosine similarities between each policy document and each framework parameter to build a codebook mapping policies to framework parameters.
While this method yields the same average similarity score, we find that splitting policy documents into sentences yields results that we can better validate with manual review.
Thus, we will not be using this script moving forward, but it is included here for reference.

"""

# basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# nlp packages
import nltk
from textblob import TextBlob
from nltk.corpus import wordnet
import re # regular expressions
from nltk.corpus import stopwords # stop words
from nltk.tokenize import word_tokenize # for word tokenization
#nltk.download('punkt') # download punkt tokenizer, only run this once
from nltk.tokenize import sent_tokenize # for sentence tokenization
from nltk.stem import WordNetLemmatizer # for stemming words
from sklearn.feature_extraction.text import CountVectorizer # for word counts
from wordcloud import WordCloud # for creating word cloud
from nltk import ngrams # for extracting phrases
from nltk.sentiment import SentimentIntensityAnalyzer # sentiment analysis
from sentence_transformers import SentenceTransformer # for semantic embeddings
from sentence_transformers import util # for semantic search

# Load standards data
data = pd.read_csv('./data/standards_with_text.csv')
print(data.head())
print(data.columns)

############################## TEXT PRE-PROCESSING ##############################
# First, process the raw text of each policy document to prepare it for analysis.

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
    text_clean = [process_text(r) for r in text_to_clean] # Outputs as list of words
    data_clean = data.copy()
    data_clean['clean_text_list'] = text_clean
    data_clean['clean_text_str'] = data_clean['clean_text_list'].apply(lambda x: ' '.join(x)) # Convert list of words back to string
    return data_clean

# Get processed text
data_clean = process_standards(data)

# Compare text output
print(data_clean[['raw_text', 'clean_text_list','clean_text_str']].head())

############################## EMBEDDINGS ##############################
# Use a pre-trained language model to convert the policy text and framework text into numerical vectors (embeddings) that capture semantic meaning.

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the policy clean text strings
data_clean['embeddings'] = data_clean['clean_text_str'].apply(lambda x: model.encode(x))

# Export updated df with clean text and embeddings to csv
data_clean.to_csv('./data/standards_clean.csv', index=False)

# Generate embeddings for the framework text
with open('./data/framework.json', 'r') as f:
    framework = json.load(f)

framework_embeddings = {}
for item in framework['framework']:
    parameter = item['parameter']
    
    # Combine definition and keywords into one description
    # This gives the model more context to understand what to look for
    keywords_text = ', '.join(item['keywords']) if item['keywords'] else ''
    combined_text = f"{item['definition']} Related terms: {keywords_text}"
    
    # Generate embedding
    embedding = model.encode(combined_text)
    
    framework_embeddings[parameter] = {
        'embedding': embedding,
        'category': item['category'],
        'subcategory': item['subcategory'],
        'definition': item['definition'],
        'keywords': item['keywords']
    }

############################## SIMILARITIES ##############################
# Calculate cosine similarities between policy document embeddings and framework parameter embeddings to build the codebook.

# Convert embeddings to numpy arrays
policy_embeddings = np.array(data_clean['embeddings'].tolist())

framework_embeddings_array = np.array([param_data['embedding'] for param_data in framework_embeddings.values()])

# Compare each policy embedding to each framework parameter embedding
similarities = model.similarity(policy_embeddings, framework_embeddings_array)

# Review similarities
similarities[1, 0]
similarities[1, 5]
similarities[1, 20]

# Create dataframe with policy titles, framework parameters, and similarity scores
framework_parameter_names = list(framework_embeddings.keys()) # Get the framework parameter names in the same order as the embeddings array
similarities_array = similarities.numpy() # Convert similarities tensor to numpy array

similarity_df = pd.DataFrame(
    similarities_array,
    columns=framework_parameter_names,
    index=data_clean['title']
)
print(similarity_df.head())

# Save similarity dataframe to CSV
similarity_df.to_csv('./data/similarity_scores.csv')

# Visualize similarity scores in order to determine optimal threshold
all_scores = similarities_array.flatten() # Flatten all similarity scores

# Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(all_scores, bins=50, edgecolor='black')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Similarity Scores')
plt.axvline(x=0.3, color='r', linestyle='--', label='Threshold=0.3')
plt.axvline(x=0.4, color='orange', linestyle='--', label='Threshold=0.4')
plt.axvline(x=0.5, color='green', linestyle='--', label='Threshold=0.5')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(all_scores)
plt.ylabel('Similarity Score')
plt.title('Box Plot of Similarity Scores')
plt.tight_layout()
plt.savefig('./plots/similarity_distribution.png')
plt.show()

# Print summary statistics
print("=== Similarity Score Statistics ===")
print(f"Mean: {np.mean(all_scores):.3f}")
print(f"Median: {np.median(all_scores):.3f}")
print(f"Std Dev: {np.std(all_scores):.3f}")
print(f"Min: {np.min(all_scores):.3f}")
print(f"Max: {np.max(all_scores):.3f}")
print(f"\nPercentiles:")
for p in [25, 50, 75, 90, 95]:
    print(f"  {p}th percentile: {np.percentile(all_scores, p):.3f}")

# Use a threshold of 0.35 to determine matches, which represents approximately the top 10% of similarity scores based on the distribution

# Refactor similarity score table according to threshold
threshold = 0.35
similarity_df_threshold = similarity_df[similarity_df >= threshold]
print(similarity_df_threshold.head())

# Convert scores above/below threshold to binary Yes/No
similarity_df_binary = similarity_df.copy()
similarity_df_binary.iloc[:, 0:] = np.where(similarity_df.iloc[:, 0:] >= threshold, 1, 0)
print(similarity_df_binary.head())
similarity_df_binary.to_csv('./data/similarity_scores_binary.csv')


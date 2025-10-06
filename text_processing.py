"""
Title: text_processing.py
Author: @dsherbini
Date: Oct 6, 2025

Processing raw text from data work standards for analysis, including text cleaning and generating embeddings.
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
nltk.download('punkt') # download punkt tokenizer
from nltk.tokenize import sent_tokenize # for sentence tokenization
from nltk.stem import WordNetLemmatizer # for stemming words
from sklearn.feature_extraction.text import CountVectorizer # for word counts
from wordcloud import WordCloud # for creating word cloud
from nltk import ngrams # for extracting phrases
from nltk.sentiment import SentimentIntensityAnalyzer # sentiment analysis
from sentence_transformers import SentenceTransformer # for semantic embeddings
from sentence_transformers import util # for semantic search

# Load standards data
data = pd.read_csv('/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/standards_with_text.csv')
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
data_clean.to_csv('/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/standards_clean.csv', index=False)

# Generate embeddings for the framework text
with open('/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/framework.json', 'r') as f:
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
similarities[0, 0]
similarities[0, 1]
similarities[0, 2]

# Create dataframe with policy titles, framework parameters, and similarity scores
framework_parameter_names = list(framework_embeddings.keys()) # Get the framework parameter names in the same order as the embeddings array
similarities_array = similarities.numpy() # Convert similarities tensor to numpy array

similarity_df = pd.DataFrame(
    similarities_array,
    columns=framework_parameter_names,
    index=data_clean['title']
)
print(similarity_df.head())

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
plt.savefig('similarity_distribution.png')
plt.show()

# Print statistics
print("=== Similarity Score Statistics ===")
print(f"Mean: {np.mean(all_scores):.3f}")
print(f"Median: {np.median(all_scores):.3f}")
print(f"Std Dev: {np.std(all_scores):.3f}")
print(f"Min: {np.min(all_scores):.3f}")
print(f"Max: {np.max(all_scores):.3f}")
print(f"\nPercentiles:")
for p in [25, 50, 75, 90, 95]:
    print(f"  {p}th percentile: {np.percentile(all_scores, p):.3f}")

# Use a threshold of 0.3 to determine matches

############################## SEMANTIC SEARCH QUERY ##############################
# Some documents are very long. To improve matching, we break them into smaller chunks and analyze those individually.
# For each policy, we search each of its smaller chunks against the framework (each parameter in the framework is a query) and extract the top 3 most relevant chunks of text for each parameter in our framework.
# We use a similarity score threshold of .3, which we deem reasonable based on the plots/summary statistics from above.

def chunk_text(text, chunk_size=100, overlap=20):
    """
    Split text into overlapping chunks of words.
    
    Parameters
    ----------
    text : (str) The text to chunk
    chunk_size : (int) Approximate number of words per chunk
    overlap : (int) Number of words to overlap between chunks
    
    Returns
    -------
    list of str : The text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    
    return chunks

def chunk_by_sentences(text, sentences_per_chunk=3, overlap_sentences=1):
    """
    Chunk text by sentences using NLTK.

    Parameters
    ----------
    text : (str) The text to chunk
    sentences_per_chink: (int) Number of sentences per chunk
    overlap_sentences : (int) Number of sentences to overlap between chunks
    
    Returns
    -------
    list of str : The text chunks

    """
    sentences = sent_tokenize(text)
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk - overlap_sentences):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        if len(chunk.strip()) > 20:  # Minimum character threshold
            chunks.append(chunk.strip())
    
    return chunks


def find_relevant_passages(policy_text, framework_embeddings, model, top_k=3):
    """
    Find the most relevant passages in a policy for each framework parameter
    
    Parameters
    ----------
    policy_text : (str) The full text of the policy document
    framework_embeddings : (dict)Dictionary with parameter names as keys and embedding data as values
    model : The sentence transformer model
    top_k : (int) Number of top (most similar) passages to return per parameter
    chunk_size : (int) Size of text chunks in words
    
    Returns
    -------
    dict : Results for each parameter
    """
    # Chunk the policy text
    chunks = chunk_by_sentences(policy_text, sentences_per_chunk=3, overlap_sentences=1)
    
    if len(chunks) == 0:
        return {}
    
    # Generate embeddings for all chunks of text (as tensors)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    
    # Pull out names of each parameter
    parameter_names = list(framework_embeddings.keys())

    # Pull out embeddings for each parameter 
    parameter_embeddings = np.array([framework_embeddings[param]['embedding'] for param in parameter_names])    
    
    # Convert parameter embeddings to tensor for semantic_search
    import torch
    parameter_embeddings = torch.tensor(parameter_embeddings)

    # Perform semantic search for all parameters at once
    search_results = util.semantic_search(
        parameter_embeddings,
        chunk_embeddings, 
        top_k=top_k
    )
    
    # Format results
    results = {}
    for idx, parameter in enumerate(parameter_names):
        results[parameter] = {
            'category': framework_embeddings[parameter]['category'],
            'subcategory': framework_embeddings[parameter]['subcategory'],
            'matches': []
        }
        
        for match in search_results[idx]:
            chunk_idx = match['corpus_id']
            score = match['score']
            
            results[parameter]['matches'].append({
                'text': chunks[chunk_idx],
                'score': score,
                'chunk_index': chunk_idx
            })
    
    return results

# Process all policies
all_results = []

print("Processing policies...")
for idx, row in data_clean.iterrows():
    policy_name = row['org']+ ': ' + row['title']
    policy_text = row['raw_text'] # Use the original raw text for chunking, in order to pull out coherent passages (as opposed to just clean words)
    #policy_text = row['clean_text_str']  # Or use cleaned text
    
    print(f"\nProcessing: {policy_name}")
    
    # Find relevant passages
    results = find_relevant_passages(
        policy_text, 
        framework_embeddings, 
        model, 
        top_k=3,  # Get top 3 passages per parameter
    )
    
    # Store results for this policy
    for parameter, data in results.items():
        for match in data['matches']:
            all_results.append({
                'policy': policy_name,
                'parameter': parameter,
                'category': data['category'],
                'subcategory': data['subcategory'],
                'text_passage': match['text'],
                'similarity_score': match['score'],
                'chunk_index': match['chunk_index']
            })
    
    print(f"Found {len(results)} parameters with relevant passages.")

# Create DataFrame with all results
results_df = pd.DataFrame(all_results)

# Filter results by similarity score threshold of .3
results_df_filtered = results_df[results_df['similarity_score'] >= 0.3]  # Apply threshold

# Save results to CSV
results_df.to_csv(
    '/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/semantic_search_results.csv',
    index=False
)

# Quick summary of results
print(f"\n{'='*80}")
print(f"Found {len(results_df)} total matches.")
print(f"Found {len(results_df_filtered)} total matches above similarity threshold of 0.3.")
print(f"Results saved to: semantic_search_results.csv")
print(f"{'='*80}")

# Display sample results
print("\n=== Sample Results ===")
print(results_df.head(10))
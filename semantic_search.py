"""
Title: semantic_search.py
Author: @dsherbini
Date: Oct 22, 2025

Description:
This script uses semantic search to query each policy document and extract relevant pieces of text.

Defining corpus: 
- Each policy document is split into sentences. These sentences represent the corpus to be searched.

Defining queries:
- We test two strategies for querying each policy document: 
    1. Each query is the paramater defintion
    2. Each query is the parameter defintion + a set of keywords related to the parameter

Semantic search process: 
- We use a pre-trained sentence transformer model to encode both the queries and the sentences from the policy documents.
- We use sbert's semantic search to find the top matching sentences for each query.
- We find that strategy 1, using definitions only, yields a higher average similarity score.

Extracting relevant text:
- We then extract the top matching sentences for all queries for each policy document.
- Full output is saved as a CSV file: 'semantic_search_full_results.csv'

Determining framework adherence: 
- We then determine whether a policy adheres to a given framework parameter by using the similarity score threshold extracted from the semantic search process, 0.35
- We define adherence as a binary variable based on whether any similarity scores of the top 5 relevant sentences matches or exceeds the threshold.
- We output these results in a CSV file: 'semantic_search_framework_adherence.csv'

"""

# basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap # for text wrapping in tables/plots
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
import torch

# Load standards data
data = pd.read_csv('./data/standards_with_text.csv')
print(data.head())
print(data.columns)

# Load framework data
with open('./data/framework.json', 'r') as f:
    framework = json.load(f)

############################## SEMANTIC SEARCH ##############################

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Put semantic search into a function and identify best query strategy (definition or definition + keywords)
def semantic_search_comparison(policy, framework, model, top_k=5, verbose=True):
    """
    Analyze a single policy document using semantic search with two query strategies: definition only, or definition + keywords.
    Use to determine optimal query strategy for extracting relevant text.
    
    Args:
        policy_text: Raw text of the policy document
        framework: Framework json containing definition and keywords
        model: Semantic search model from SentenceTransformer
        top_k: Number of top sentences to retrieve (default: 5)
        verbose: Whether to print results (default: True)
    
    Returns:
        Dictionary containing:
            - avg_scores_def_only: List of average scores for definition-only strategy
            - avg_scores_with_keywords: List of average scores for definition+keywords strategy
            - corpus: List of sentences from the document
            - overall_avg_def_only: Overall average score for definition-only
            - overall_avg_with_keywords: Overall average score for definition+keywords
    """
    
    # If verbose = true, print policy document text
    if verbose:
        print(f"Policy Document: {policy['title']}")
        print(policy['raw_text'])
        print("\n" + "="*80)
    
    # Split policy document into sentences -- this is the corpus
    corpus = sent_tokenize(policy['raw_text'])
    
    # If verbose, print corpus info
    if verbose:
        print(f"\nDocument split into {len(corpus)} sentences")
        print("First 5 sentences:")
        for i, sent in enumerate(corpus[:5], 1):
            print(f"  {i}. {sent}")
        print("\n" + "="*80)
    
    # Encode corpus/get corpus embeddings
    corpus_embeddings = model.encode_document(corpus, convert_to_tensor=True)
    
    # Prepare queries -- both definition only and definition + keywords
    queries_def_only = [item['definition'] for item in framework['framework']]
    query_with_keywords = [
        f"{item['definition']}. Related terms: {', '.join(item['keywords']) if item['keywords'] else ''}" 
        for item in framework['framework']
    ]
    
    if verbose:
        print(f"\nTotal queries: {len(queries_def_only)}")
        print("\n" + "="*80)
    
    # Determine actual top_k based on corpus size; if corpus has fewer sentences than top_k, adjust accordingly
    actual_top_k = min(top_k, len(corpus))
    
    # Storage for results
    avg_scores_def_only = []
    avg_scores_with_keywords = []
    
    # Strategy 1: Definition only
    if verbose:
        print("\n STRATEGY 1: DEFINITION ONLY")
        print("="*80)
    
    for i, query in enumerate(queries_def_only):
        query_embedding = model.encode_query(query, convert_to_tensor=True)
        similarity_scores = model.similarity(query_embedding, corpus_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=actual_top_k)
        
        if verbose:
            print(f"\nQuery {i+1}: {query}")
            print(f"Top {actual_top_k} most similar sentences:")
            for rank, (score, idx) in enumerate(zip(scores, indices), 1):
                print(f"  {rank}. [Score: {score:.4f}] {corpus[idx]}")
        
        # Calculate and store average score
        avg_score = torch.mean(scores).item()
        avg_scores_def_only.append(avg_score)
        
        if verbose:
            print(f"Average similarity score: {avg_score:.4f}")
    
    # Strategy 2: Definition + keywords
    if verbose:
        print("\n\nSTRATEGY 2: DEFINITION + KEYWORDS")
        print("="*80)
    
    for i, query in enumerate(query_with_keywords):
        query_embedding = model.encode_query(query, convert_to_tensor=True)
        similarity_scores = model.similarity(query_embedding, corpus_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=actual_top_k)
        
        if verbose:
            print(f"\nQuery {i+1}: {query}")
            print(f"Top {actual_top_k} most similar sentences:")
            for rank, (score, idx) in enumerate(zip(scores, indices), 1):
                print(f"  {rank}. [Score: {score:.4f}] {corpus[idx]}")
        
        # Calculate and store average score
        avg_score = torch.mean(scores).item()
        avg_scores_with_keywords.append(avg_score)
        
        if verbose:
            print(f"Average similarity score: {avg_score:.4f}")
    
    # Calculate overall averages
    overall_avg_def_only = sum(avg_scores_def_only) / len(avg_scores_def_only)
    overall_avg_with_keywords = sum(avg_scores_with_keywords) / len(avg_scores_with_keywords)
    
    if verbose:
        print("\n\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        print(f"\nDefinition Only - Overall Average: {overall_avg_def_only:.4f}")
        print(f"Definition + Keywords - Overall Average: {overall_avg_with_keywords:.4f}")
        print(f"Difference: {overall_avg_with_keywords - overall_avg_def_only:+.4f}")
    
    return {
        'avg_scores_def_only': avg_scores_def_only,
        'avg_scores_with_keywords': avg_scores_with_keywords,
        'corpus': corpus,
        'corpus_size': len(corpus),
        'overall_avg_def_only': overall_avg_def_only,
        'overall_avg_with_keywords': overall_avg_with_keywords,
        'difference': overall_avg_with_keywords - overall_avg_def_only
    }


# Analyze a single document -- check verbose output against previous test
policy = data.iloc[1]
results = semantic_search_comparison(policy, framework, model, top_k=5, verbose=True)

# Access results
print(f"\nDefinition only scores: {results['avg_scores_def_only']}")
print(f"Keywords scores: {results['avg_scores_with_keywords']}")
print(f"Overall difference: {results['difference']:.4f}")

# Analyze all policy documents -- without verbose output
all_results = []
for i, row in data.iterrows():
    print(f"Processing document {i+1}/{len(data)}: {row['title']}")
    result = semantic_search_comparison(row, framework, model, top_k=5, verbose=False)
    result['doc_id'] = i
    all_results.append(result)

# Compare scores across all documents between the two strategies
overall_def = sum(r['overall_avg_def_only'] for r in all_results) / len(all_results)
overall_keywords = sum(r['overall_avg_with_keywords'] for r in all_results) / len(all_results)
print(f"\nAcross all {len(all_results)} documents:")
print(f"Definition only average: {overall_def:.4f}")
print(f"Keywords average: {overall_keywords:.4f}")

# Visualize similarity scores for the two strategies
# Extract scores from all results
all_def_only_scores = []
all_with_keywords_scores = []

for result in all_results:
    all_def_only_scores.extend(result['avg_scores_def_only'])
    all_with_keywords_scores.extend(result['avg_scores_with_keywords'])

# Calculate averages
avg_def_only = np.mean(all_def_only_scores)
avg_with_keywords = np.mean(all_with_keywords_scores)

# Now plot
plt.figure(figsize=(12, 5))

# Plot 1: Histogram of definition-only scores
plt.subplot(1, 2, 1)
plt.hist(all_def_only_scores, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(avg_def_only, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_def_only:.4f}')
plt.title('Distribution of Similarity Scores\n(Definition Only)')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.legend()

# Plot 2: Histogram of definition+keywords scores
plt.subplot(1, 2, 2)
plt.hist(all_with_keywords_scores, bins=30, edgecolor='black', alpha=0.7, color='orange')
plt.axvline(avg_with_keywords, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_with_keywords:.4f}')
plt.title('Distribution of Similarity Scores\n(Definition + Keywords)')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig('./plots/similarity_scores_distribution.png')
plt.show()

# Results: Definition only strategy yields higher average scores across all documents, so we will use that going forward.
# We get an average similarity score of 0.3575, which we will use as our similarity threshold for determining framework adherence below.

# Save semantic search results, using definitions only as queries
def semantic_search_extraction(policy, framework, model, top_k=5, verbose=True):
    """
    Extract relevant sentences from a policy document using semantic search with definition-only queries.
    
    Args:
        policy_text: Raw text of the policy document
        framework: Framework json containing definitions, which will be used as queries
        model: Semantic search model from SentenceTransformer
        top_k: Number of top sentences to retrieve (default: 5)
        verbose: Whether to print results (default: True)
    
    Returns:
        Dictionary containing:
            - policy_title: Title of the policy document
            - results: List of dicts, each containing query and its top relevant sentences

    """
    
    # If verbose = true, print policy document text
    if verbose:
        print(f"Policy Document: {policy['title']}")
        print(policy['raw_text'])
        print("\n" + "="*80)
    
    # Split policy document into sentences -- this is the corpus
    corpus = sent_tokenize(policy['raw_text'])
    
    # If verbose, print corpus info
    if verbose:
        print(f"\nDocument split into {len(corpus)} sentences")
        print("First 5 sentences:")
        for i, sent in enumerate(corpus[:5], 1):
            print(f"  {i}. {sent}")
        print("\n" + "="*80)
    
    # Encode corpus/get corpus embeddings
    corpus_embeddings = model.encode_document(corpus, convert_to_tensor=True)
    
    # Prepare queries
    queries = [item['definition'] for item in framework['framework']]
    
    if verbose:
        print(f"\nTotal queries: {len(queries)}")
        print("\n" + "="*80)
    
    # Determine actual top_k based on corpus size; if corpus has fewer sentences than top_k, adjust accordingly
    actual_top_k = min(top_k, len(corpus))

    # Storage for results
    extraction_results = []
    
    # Run semantic search for each query
    if verbose:
        print("\n QUERYING...")
        print("="*80)
    
    for i, query in enumerate(queries):
        query_embedding = model.encode_query(query, convert_to_tensor=True)
        similarity_scores = model.similarity(query_embedding, corpus_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=actual_top_k)
        
        # Store sentences with their scores
        relevant_sentences = []
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            relevant_sentences.append({
                'rank': rank,
                'score': score.item(),
                'sentence': corpus[idx],
                'sentence_index': idx.item()
            })
        
        # Store query and its results
        query_result = {
            'query_id': i,
            'query': query,
            'parameter': framework['framework'][i]['parameter'],
            'category': framework['framework'][i]['category'],
            'relevant_sentences': relevant_sentences
        }
        extraction_results.append(query_result)
        
        if verbose:
            print(f"\nQuery {i+1}: {query}")
            print(f"Top {actual_top_k} most similar sentences:")
            for sent_info in relevant_sentences:
                print(f"  {sent_info['rank']}. [Score: {sent_info['score']:.4f}] {sent_info['sentence']}")
    
    return {
        'policy_title': policy['title'],
        'corpus_size': len(corpus),
        'results': extraction_results
    }

# Test extraction function on one document
policy = data.iloc[1]
extracted_data = semantic_search_extraction(policy, framework, model, top_k=5, verbose=True)
print(f"\nPolicy: {extracted_data['policy_title']}")
print(f"Results: {(extracted_data['results'])}")

# Get sentences for a specific query
first_query_results = extracted_data['results'][0]
print(f"\nQuery: {first_query_results['query']}")
print(f"Parameter: {first_query_results['parameter']}")
for sentence in first_query_results['relevant_sentences']:
    print(f" - (Score: {sentence['score']:.4f}) --> {sentence['sentence']} ")

# Save extraction results for all documents into a csv file
all_extraction_results = []
for i, row in data.iterrows():
    print(f"Extracting from document {i+1}/{len(data)}: {row['title']}")
    extraction_result = semantic_search_extraction(row, framework, model, top_k=5, verbose=False)
    
    # Flatten results for saving to CSV
    for query_result in extraction_result['results']:
        for sentence_info in query_result['relevant_sentences']:
            all_extraction_results.append({
                'policy_title': extraction_result['policy_title'],
                'corpus_size': extraction_result['corpus_size'],
                'query_id': query_result['query_id'],
                'parameter': query_result['parameter'],
                'category': query_result['category'],
                'query': query_result['query'],
                'sentence_rank': sentence_info['rank'],
                'sentence_score': sentence_info['score'],
                'sentence': sentence_info['sentence'],
                'sentence_index': sentence_info['sentence_index']
            })

# Convert to dataframe and save to CSV
extraction_df = pd.DataFrame(all_extraction_results)
extraction_df.to_csv('./data/semantic_search_extractions.csv', index=False)

# Determine framework adherence based on similarity score threshold
threshold = 0.3575
adherence_results = []

for i, row in data.iterrows():
    policy_title = row['title']
    policy_extractions = extraction_df[extraction_df['policy_title'] == policy_title]
    
    for param in framework['framework']:
        parameter_name = param['parameter']
        param_extractions = policy_extractions[policy_extractions['parameter'] == parameter_name]
        
        # Check if any of the top 5 sentences exceed the threshold
        adheres = int((param_extractions['sentence_score'] >= threshold).any())
        
        adherence_results.append({
            'policy_title': policy_title,
            'parameter': parameter_name,
            'category': param['category'],
            'adheres': adheres
        })

# Convert to dataframe
adherence_df = pd.DataFrame(adherence_results)

# Pivot the dataframe: policies as rows, parameters as columns
adherence_pivot = adherence_df.pivot(
    index='policy_title',
    columns='parameter',
    values='adheres'
)

# Reset index to make policy_title a regular column
adherence_pivot = adherence_pivot.reset_index()

# Merge with original policy metadata
columns_to_add = ['org','doc_type','org_type', 'geography', 'date']
data_subset = data[['title'] + columns_to_add].copy() # Grab columns as a copy

# Merge the data
adherence_pivot = data_subset.merge(
    adherence_pivot,
    left_on='title',
    right_on='policy_title',
    how='left'
)

# Drop the duplicate 'policy_title' column
adherence_pivot = adherence_pivot.drop(columns=['policy_title'])

# Save to CSV
adherence_pivot.to_csv('./data/semantic_search_adherence.csv', index=False)

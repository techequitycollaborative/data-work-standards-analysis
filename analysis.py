"""
Title: analysis.py
Author: @dsherbini
Date: Jan 5, 2026

For our analysis, we look at:
1. Frequency of parameter queries per policy document 
2. Frequency of parameter queries overall (i.e. total mentions across all documents)
3. Distribution of parameter queries across different document variables
4. Adherence of each document to the framework / by document variables
5. Top 10 most/least frequently mentioned parameters
6. Thematic analysis of top and bottom most mentioned parameters
7. Breadth/Consensus x Depth/Emphasis 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# For word clouds
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

# For phrase extraction
import nltk
from nltk import ngrams
from nltk.corpus import stopwords as nltk_stopwords

# Download nltk data -- run only once
#nltk.download('punkt')
#nltk.download('stopwords')


# Load results from manual review
data = pd.read_csv('./data/review_results.csv')
print(data.head())
print(data.columns)

# Load standards and framework data (for merging later)
standards = pd.read_csv('./data/standards.csv')
framework = pd.read_csv('./data/framework.csv')


#####  1. Frequency of parameter queries per policy document ##### 
param_columns = ['primary_paramater', 'alt_parameter_1', 'alt_parameter_2', 
                 'alt_parameter_3', 'alt_parameter_4']

# Melt the dataframe to get all parameter values in one column
melted = data.melt(
    id_vars=['policy_title','sentence'], 
    value_vars=param_columns,
    value_name='parameter_value'
)

# Remove null values
melted = melted[melted['parameter_value'].notna()]

# Merge with shorthand framework parameters
shorthand_params = framework[['Parameter','Definition']].copy()
shorthand_params = shorthand_params.rename(columns={'Definition': 'parameter_value', 'Parameter': 'parameter'})
melted = melted.merge(
    shorthand_params,
    on='parameter_value',
    how='left'
)

# Count number of times each query appears per policy document
summary = melted.groupby(['policy_title', 'parameter']).size().reset_index(name='count')

# Pivot to get parameter values as columns
result = summary.pivot(index='policy_title', columns='parameter', values='count').fillna(0).reset_index()

# Put columns in same order as original framework params
desired_order = framework['Parameter'].tolist()
column_order = ['policy_title'] + desired_order
result = result.reindex(columns=column_order, fill_value=0) # Reindex to include all columns, filling missing ones with 0
result.iloc[:, 1:] = result.iloc[:, 1:].astype(int) # Convert counts to integers (skip policy_title column)

# Merge with original policy metadata + corpus size
columns_to_add = ['org','doc_type','org_type', 'worker_focus', 'geography', 'date']
standards_data = standards[['title'] + columns_to_add].copy()

corpus_size = data[['policy_title','corpus_size']].drop_duplicates() # Get unique corpus sizes

# Combine all metadata
metadata = standards_data.merge(
    corpus_size,
    left_on='title',
    right_on='policy_title',
    how='left'
).drop(columns=['policy_title'])

# Merge metadata with result df
param_frequency_by_policy = metadata.merge(
    result,
    left_on='title',
    right_on='policy_title',
    how='left'
).drop(columns=['policy_title'])

# Save to CSV
param_frequency_by_policy.to_csv('./data/param_frequency_by_policy.csv', index=False)

##### 2. Frequency of parameter queries overall ##### 
param_frequency_overall = melted.groupby('parameter').agg(
    total_appearances=('parameter', 'size'),
    num_docs=('policy_title', 'nunique')
).reset_index()

# Rename columns
param_frequency_overall.columns = ['parameter', 'total_appearances', 'num_docs']

# Reorder rows based on framework order
param_frequency_overall = param_frequency_overall.set_index('parameter')
param_frequency_overall = param_frequency_overall.reindex(desired_order)
param_frequency_overall= param_frequency_overall.reset_index()

# Fill NaN values with 0
param_frequency_overall[['total_appearances', 'num_docs']] = param_frequency_overall[['total_appearances', 'num_docs']].fillna(0).astype(int)

# Sort by total appearances -- optional
#param_frequency_overall = param_frequency_overall.sort_values(by='total_appearances', ascending=False).reset_index(drop=True)

# Merge with other framework info
framework_info = framework[['Parameter','Category', 'Subcategory', 'Definition']].rename(columns={'Parameter': 'parameter','Category':'category', 'Subcategory':'subcategory', 'Definition':'definition'})
param_frequency_overall = framework_info.merge(
    param_frequency_overall,
    on='parameter',
    how='left'
)

# Save to CSV
param_frequency_overall.to_csv('./data/param_frequency_overall.csv', index=False)

# Plots of parameter frequency

# Heatmaps
# By total appearances
plot_data_1 = param_frequency_overall.sort_values(by='total_appearances', ascending=False)

plt.figure(figsize=(12, 8))
sns.heatmap(
    plot_data_1[['total_appearances']].set_index(plot_data_1['parameter']),
    annot=True,
    fmt='d',
    cmap='viridis',
)
plt.ylabel('')
plt.xlabel('Total Appearances of Parameter Across Full Corpus')
plt.title('Parameter Frequency by Total Appearances')
plt.tight_layout()
plt.savefig('./plots/param_frequency_by_total_appearances.png')
plt.show()

# By number of documents
plot_data_2 = param_frequency_overall.sort_values(by='num_docs', ascending=False)

plt.figure(figsize=(12, 8))
sns.heatmap(
    plot_data_2[['num_docs']].set_index(plot_data_2['parameter']),
    annot=True,
    fmt='d',
    cmap='viridis',
)
plt.ylabel('')
plt.xlabel('Number of Documents with Parameter Present (Max 14)')
plt.title('Parameter Frequency by Number of Documents')
plt.tight_layout()
plt.savefig('./plots/param_frequency_by_num_docs.png')
plt.show()

# Bar plots
# Total appearances
plt.figure(figsize=(14, 7))
sns.barplot(
    data=plot_data_1,
    y='parameter',
    x='total_appearances',
    hue='total_appearances',
    palette='viridis'
)
plt.ylabel('')
plt.xlabel('Total Appearances of Parameter Across Full Corpus')
plt.title('Parameter Frequency by Total Appearances')
plt.tight_layout()
plt.savefig('./plots/param_total_appearances_barplot.png')
plt.show()

# Number of documents
plt.figure(figsize=(14, 7))
sns.barplot(
    data=plot_data_2,
    y='parameter',
    x='num_docs',
    hue='num_docs',
    palette='viridis'
)
plt.ylabel('')
plt.xlabel('Number of Documents with Parameter Present (Max 14)')
plt.title('Parameter Frequency by Number of Documents')
plt.tight_layout()
plt.savefig('./plots/param_num_docs_barplot.png')
plt.show()

# Focus on subcategory frequencies
subcategory_frequency = param_frequency_overall.groupby('subcategory').agg(
    total_appearances=('total_appearances', 'sum'),
    num_docs=('num_docs', 'mean')
).reset_index()

print(subcategory_frequency)
subcategory_frequency.to_csv('./data/subcategory_frequency_overall.csv', index=False)

# Plot subcategories by total appearances and number of docs
fig, ax = plt.subplots(figsize=(20, 14))  # Increased from (18, 14)

scatter = ax.scatter(
    subcategory_frequency['num_docs'], 
    subcategory_frequency['total_appearances'],
    s=200,  # Larger points
    alpha=0.6,
    c=subcategory_frequency['num_docs'],
    cmap='viridis',
    edgecolors='black',
    linewidth=0.8
)

# Larger fonts for readability
for idx, row in subcategory_frequency.iterrows():
    ax.annotate(
        row['subcategory'],
        (row['num_docs'], row['total_appearances']),
        fontsize=16,  # Increased from 7
        alpha=0.9,
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray', linewidth=1.5)
    )

ax.set_xlabel('Number of Documents Mentioning Parameter (max 14)', fontsize=18)  # Larger
ax.set_ylabel('Total Mentions Across All Documents', fontsize=18)  # Larger
ax.set_title('Subcategory Coverage: Breadth vs. Depth', fontsize=22, fontweight='bold', pad=20)  # Larger
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

# Larger tick labels
ax.tick_params(axis='both', which='major', labelsize=14)

plt.colorbar(scatter, label='Number of Documents').ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('./plots/subcategory_coverage_scatter.png', bbox_inches='tight', dpi=300)  # High DPI
plt.show()


##### 3. Distribution of parameter queries across different document variables #####

# Select document metadata columns and parameter columns
metadata_cols = ['title', 'org', 'doc_type', 'org_type', 'worker_focus', 'geography', 'date', 'corpus_size']
param_cols = [col for col in param_frequency_by_policy.columns if col not in metadata_cols]

### Document type
# Group by doc_type and sum parameter frequencies
param_by_doctype = param_frequency_by_policy.groupby('doc_type')[param_cols].sum()

# Normalize by row (each doc type sums to 100%)
# We normalize in order to see how much emphasis each document type gives to each parameter
param_by_doctype_normalized = param_by_doctype.div(param_by_doctype.sum(axis=1), axis=0) * 100

plt.figure(figsize=(20, 8))
sns.heatmap(
    param_by_doctype_normalized.T, # .T = transpose so parameters are rows and doc type are columns
    annot=True,
    fmt='.1f',
    cmap='YlOrRd',
    cbar_kws={'label': 'Percentage (%)'}
)
plt.title('Parameter Distribution by Document Type (Normalized)')
plt.xlabel('')
plt.ylabel('')
#plt.tight_layout()
plt.savefig('./plots/param_by_doctype_heatmap.png', bbox_inches='tight')
plt.show()

### Org type
param_by_orgtype = param_frequency_by_policy.groupby('org_type')[param_cols].sum()
param_by_orgtype_normalized = param_by_orgtype.div(param_by_orgtype.sum(axis=1), axis=0) * 100

# Create heatmap
plt.figure(figsize=(20, 8))
sns.heatmap(
    param_by_orgtype_normalized.T,
    annot=True,
    fmt='g',
    cmap='YlOrRd',
    cbar_kws={'label': 'Count'}
)
plt.title('Parameter Distribution by Org Type (Normalized)')
plt.xlabel('')
plt.ylabel('')
#plt.tight_layout()
plt.savefig('./plots/param_by_orgtype_heatmap.png', bbox_inches='tight')
plt.show()

### Worker focus
param_by_workerfocus = param_frequency_by_policy.groupby('worker_focus')[param_cols].sum()
param_by_workerfocus_normalized = param_by_workerfocus.div(param_by_workerfocus.sum(axis=1), axis=0) * 100

# Create heatmap
plt.figure(figsize=(20, 8))
sns.heatmap(
    param_by_workerfocus_normalized.T,
    annot=True,
    fmt='g',
    cmap='YlOrRd',
    cbar_kws={'label': 'Count'}
)
plt.title('Parameter Distribution by Worker Focus (Normalized)')
plt.xlabel('')
plt.ylabel('')
#plt.tight_layout()
plt.savefig('./plots/param_by_workerfocus_heatmap.png', bbox_inches='tight')
plt.show()

### Geography
param_by_geography = param_frequency_by_policy.groupby('geography')[param_cols].sum()
param_by_geography_normalized = param_by_geography.div(param_by_geography.sum(axis=1), axis=0) * 100

# Create heatmap
plt.figure(figsize=(20, 8))
sns.heatmap(
    param_by_geography_normalized.T,
    annot=True,
    fmt='g',
    cmap='YlOrRd',
    cbar_kws={'label': 'Count'}
)
plt.title('Parameter Distribution by Geography (Normalized)')
plt.xlabel('')
plt.ylabel('')
#plt.tight_layout()
plt.savefig('./plots/param_by_geography_heatmap.png', bbox_inches='tight')
plt.show()

##### 4. Adherence of each document to the framework / by document variables #####

# Calculate "adherence score" - i.e. how well each document adheres to the framework
# Calculated as number of parameters mentioned / total number of parameters (41)
metadata_cols = ['title', 'org', 'doc_type', 'org_type', 'worker_focus', 'geography', 'date', 'corpus_size']
param_cols = [col for col in param_frequency_by_policy.columns if col not in metadata_cols]

# Total parameters mentioned (count non-zero)
param_frequency_by_policy['total_params_mentioned'] = (param_frequency_by_policy[param_cols] > 0).sum(axis=1)

# Total mentions across all parameters
param_frequency_by_policy['total_mentions'] = param_frequency_by_policy[param_cols].sum(axis=1)

# Proportion of framework covered (out of total possible parameters)
param_frequency_by_policy['framework_coverage'] = param_frequency_by_policy['total_params_mentioned'] / len(param_cols)

# Summary of adherence scores by document
adherence_summary = param_frequency_by_policy[['title', 'total_params_mentioned', 'total_mentions', 'framework_coverage']]
print(adherence_summary)

# Get avg params mentioned, total mentions, framework coverage
adherence_avg = adherence_summary.agg({
    'total_params_mentioned': 'mean',
    'total_mentions': 'mean',
    'framework_coverage': 'mean'
}).reset_index().rename(columns={0: 'average_value', 'index': 'metric'})

print("\nAdherence Averages Per Document:")
print(adherence_avg)


# Save to CSV
adherence_summary.to_csv('./data/adherence_summary_by_policy.csv', index=False)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Framework coverage by doc type
sns.barplot(data=param_frequency_by_policy, x='doc_type', y='framework_coverage', ax=axes[0,0], hue='doc_type', palette='Set2')
axes[0,0].set_title('Framework Coverage by Document Type')
axes[0,0].set_ylabel('Proportion of Framework Covered')
axes[0,0].set_xlabel('')
axes[0,0].tick_params(axis='x', rotation=45)

# By org type
sns.barplot(data=param_frequency_by_policy, x='org_type', y='framework_coverage', ax=axes[0,1], hue='org_type', palette='Set3')
axes[0,1].set_title('Framework Coverage by Organization Type')
axes[0,1].set_ylabel('Proportion of Framework Covered')
axes[0,1].set_xlabel('')
axes[0,1].tick_params(axis='x', rotation=45)

# By worker focus
sns.barplot(data=param_frequency_by_policy, x='worker_focus', y='framework_coverage', ax=axes[0,2], hue='worker_focus', palette='viridis')
axes[0,2].set_title('Framework Coverage by Worker Focus')
axes[0,2].set_ylabel('Proportion of Framework Covered')
axes[0,2].set_xlabel('')
axes[0,2].tick_params(axis='x', rotation=45)

# By geography
sns.barplot(data=param_frequency_by_policy, x='geography', y='framework_coverage', ax=axes[1,0], hue='geography', palette='magma')
axes[1,0].set_title('Framework Coverage by Geography')
axes[1,0].set_ylabel('Proportion of Framework Covered')
axes[1,0].set_xlabel('')
axes[1,0].tick_params(axis='x', rotation=45)

# By date
param_frequency_by_policy['date_dt'] = pd.to_datetime(param_frequency_by_policy['date'], errors='coerce') # convert to datetime
sns.scatterplot(data=param_frequency_by_policy.dropna(subset=['date_dt']), 
                x='date_dt', y='framework_coverage', 
                s=200, ax=axes[1,1], color='steelblue')  # Single color
axes[1,1].set_title('Framework Coverage Over Time')
axes[1,1].set_xlabel('')
axes[1,1].set_ylabel('Proportion of Framework Covered')
axes[1,1].tick_params(axis='x', rotation=45)

# By document length
sns.scatterplot(data=param_frequency_by_policy, x='corpus_size', y='framework_coverage', 
                s=200, ax=axes[1,2], color='steelblue')  # Single color
axes[1,2].set_title('Framework Coverage vs Document Length')
axes[1,2].set_xlabel('Document Length (in sentences)')
axes[1,2].set_ylabel('Proportion of Framework Covered')

plt.tight_layout()
plt.savefig('./plots/framework_adherence_analysis.png', dpi=300)
plt.show()


##### 5. Top 10 most/least frequently mentioned parameters #####

# Revisit overall parameter frequency data
# Add average mentions per document
param_frequency_overall['avg_mentions_per_doc'] = param_frequency_overall['total_appearances'] / param_frequency_overall['num_docs']

# Top 10 by document breadth (PRIMARY)
top_10_breadth = param_frequency_overall.nlargest(10, 'num_docs')
print("Top 10 Most Widely Adopted Parameters:")
print(top_10_breadth)
top_10_breadth.to_csv('./data/top_10_most_adopted_parameters.csv', index=False)

# Plot top 10
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=top_10_breadth,
    y='parameter',
    x='num_docs',
    ax=ax
)
ax.bar_label(ax.containers[0])  # Adds labels to bars
labels = [textwrap.fill(label.get_text(), width=50) for label in ax.get_yticklabels()] # Wrap y-axis labels
ax.set_yticklabels(labels)
plt.ylabel('')
plt.xlabel('Number of Documents with Parameter Present (Max 14)')
plt.title('Top 10 Most Widely Adopted Parameters')
sns.despine() # Remove plot borders
plt.tight_layout()
plt.savefig('./plots/top_10_most_adopted_parameters.png')

# Bottom 10 by document breadth (PRIMARY)
bottom_10_breadth = param_frequency_overall.nsmallest(10, 'num_docs')
print("\nTop 10 Least Adopted Parameters:")
print(bottom_10_breadth)
bottom_10_breadth.to_csv('./data/bottom_10_least_adopted_parameters.csv', index=False)

# Plot bottom 10
plt.figure(figsize=(10, 6))
sns.barplot(
    data=bottom_10_breadth,
    y='parameter',
    x='num_docs',
)
labels = [textwrap.fill(label.get_text(), width=50) for label in ax.get_yticklabels()] # Wrap y-axis labels
ax.set_yticklabels(labels)
plt.ylabel('')
plt.xlabel('Number of Documents with Parameter Present (Max 14)')
plt.title('Bottom 10 Parameters - i.e. Least Adopted Parameters')
sns.despine() # Remove plot borders
plt.tight_layout()
plt.savefig('./plots/bottom_10_least_adopted_parameters.png')

# Top 10 by total mentions (SECONDARY - for depth analysis, to glean emphasis (i.e. if a document mentions a parameter a lot))
top_10_depth = param_frequency_overall.nlargest(10, 'total_appearances')
print("\nTop 10 Most Frequently Discussed Parameters:")
print(top_10_depth)
top_10_depth.to_csv('./data/top_10_most_frequent_parameters.csv', index=False)

# Plot top 10 by total mentions
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_10_depth,
    y='parameter',
    x='num_docs',
)
labels = [textwrap.fill(label.get_text(), width=50) for label in ax.get_yticklabels()] # Wrap y-axis labels
ax.set_yticklabels(labels)
plt.ylabel('')
plt.xlabel('Total Appearances of Parameter Across Full Corpus')
plt.title('Top 10 Parameters - Most Total Mentions')
sns.despine() # Remove plot borders
plt.tight_layout()
plt.savefig('./plots/top_10_most_mentioned_parameters.png')

# Get average mentions across all parameters
average_mentions_per_doc = param_frequency_overall['avg_mentions_per_doc'].mean()
print(average_mentions_per_doc)

# Make histogram of average mentions per document
plt.figure(figsize=(10, 6))
sns.histplot(
    data=param_frequency_overall,
    x='avg_mentions_per_doc',
    bins=15,
    kde=True,
    color='steelblue'
)
plt.axvline(average_mentions_per_doc, color='red', linestyle='--', label='Overall Average')
plt.xlabel('Average Mentions per Document')
plt.title('Distribution of Average Mentions per Document Across Parameters')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/avg_mentions_per_doc_distribution.png', dpi=300)
plt.show()



##### 6. Thematic analysis of top and bottom most mentioned parameters #####

# Focus on top three parameters for word cloud/theme analysis
params = {
    'Freedom of Association': melted[melted['parameter'] == 'Freedom of association and fair representation (democratically determined working conditions, worker voice mechanisms)'],
    'Health and Safety': melted[melted['parameter'] == 'Health and safety risks'],
    'Wages': melted[melted['parameter'] == 'Wages']
}

# Create combined stopwords set
stop_words = set(STOPWORDS) | set(nltk_stopwords.words('english'))

for param_name, param_data in params.items():
    print(f"\n{'='*60}")
    print(f"Processing: {param_name}")
    print(f"Found {len(param_data)} sentences")
    print(f"{'='*60}")
    
    # Combine all sentences into single text
    text = ' '.join(param_data['sentence'].astype(str))
    text_lower = text.lower()

    # Individual words
    print(f"\n=== Top 20 Individual Words: {param_name} ===")

    # Filter words
    filtered_words = [word for word in text_lower.split() 
                      if word not in stop_words and len(word) > 3]
    word_freq = Counter(filtered_words)

    # Print top words
    for word, count in word_freq.most_common(20):
        print(f"{word}: {count}")

    # Create word cloud
    wordcloud = WordCloud(
        width=1200, 
        height=600,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud: {param_name} (Individual Words)', fontsize=18)
    plt.tight_layout()
    
    # Create clean filename
    filename = param_name.lower().replace(' ', '_').replace('&', 'and')
    plt.savefig(f'./plots/{filename}_words_wordcloud.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Phrases
    print(f"\n=== Top 20 Two-Word Phrases: {param_name} ===")

    # Tokenize and filter
    tokens = nltk.word_tokenize(text_lower)
    filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 2]

    # Create bigrams
    bigrams = list(ngrams(filtered_tokens, 2))
    bigram_phrases = [' '.join(gram) for gram in bigrams]
    bigram_freq = Counter(bigram_phrases)

    # Print top phrases
    for phrase, count in bigram_freq.most_common(20):
        print(f"{phrase}: {count}")

    # Create word cloud
    wordcloud = WordCloud(
        width=1200, 
        height=600,
        background_color='white',
        colormap='plasma'
    ).generate_from_frequencies(dict(bigram_freq.most_common(50)))

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Key Themes: {param_name} (Two-Word Phrases)', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'./plots/{filename}_themes_wordcloud.png', bbox_inches='tight', dpi=300)
    plt.show()

print("\n" + "="*60)
print("All word clouds generated successfully!")
print("="*60)

# Look at sentences for top parameters
freedom_sentences = params['Freedom of Association'][['policy_title', 'sentence']].drop_duplicates()

##### 7. Breadth/Consensus x Depth/Emphasis #####

param_coverage = param_frequency_overall[['parameter', 'total_appearances', 'num_docs']].copy()

param_coverage['parameter_short'] = param_coverage['parameter'].apply(
    lambda x: '\n'.join(textwrap.wrap(x, width=25))
)

# Larger figure size for presentations
fig, ax = plt.subplots(figsize=(20, 14))  # Increased from (18, 14)

scatter = ax.scatter(
    param_coverage['num_docs'], 
    param_coverage['total_appearances'],
    s=200,  # Larger points
    alpha=0.6,
    c=param_coverage['num_docs'],
    cmap='viridis',
    edgecolors='black',
    linewidth=0.8
)

# Larger fonts for readability
for idx, row in param_coverage.iterrows():
    ax.annotate(
        row['parameter_short'], 
        (row['num_docs'], row['total_appearances']),
        fontsize=10,  # Increased from 7
        alpha=0.9,
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray', linewidth=1.5)
    )

ax.set_xlabel('Number of Documents Mentioning Parameter (max 14)', fontsize=18)  # Larger
ax.set_ylabel('Total Mentions Across All Documents', fontsize=18)  # Larger
ax.set_title('Parameter Coverage: Breadth vs. Depth', fontsize=22, fontweight='bold', pad=20)  # Larger
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

# Larger tick labels
ax.tick_params(axis='both', which='major', labelsize=14)

plt.colorbar(scatter, label='Number of Documents').ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('./plots/parameter_coverage_scatter.png', bbox_inches='tight', dpi=300)  # High DPI
plt.show()
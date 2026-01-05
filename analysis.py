"""
Title: analysis.py
Author: @dsherbini
Date: Jan 5, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results from manual review
data = pd.read_csv('./data/review_results.csv')
print(data.head())
print(data.columns)

# For our analysis, we want to look at:
# 1. Frequency of parameter queries overall
# 2. Frequency of parameter queries per policy document
# 3. Unique parameters queries per document
# 4. Distribution of parameter queries across different document types

# 1. Frequency of parameter queries overall
param_columns = ['primary_paramater', 'alt_parameter_1', 'alt_parameter_2', 
                 'alt_parameter_3', 'alt_parameter_4']

# Melt the dataframe to get all parameter values in one column
melted = data.melt(
    id_vars=['policy_title'], 
    value_vars=param_columns,
    value_name='parameter_value'
)

# Remove null values
melted = melted[melted['parameter_value'].notna()]

# Count number of times each query appears per policy document
summary = melted.groupby(['policy_title', 'parameter_value']).size().reset_index(name='count')

# Pivot to get parameter values as columns
result = summary.pivot(index='policy_title', columns='parameter_value', values='count').fillna(0).reset_index()

# Put columns in same order as original framework params
framework = pd.read_csv('./data/framework.csv')
desired_order = framework['Definition'].tolist()
column_order = ['policy_title'] + desired_order
result = result.reindex(columns=column_order, fill_value=0) # Reindex to include all columns, filling missing ones with 0
result.iloc[:, 1:] = result.iloc[:, 1:].astype(int) # Convert counts to integers (skip policy_title column)

# Merge with original policy metadata + corpus size
standards = pd.read_csv('./data/standards.csv')
columns_to_add = ['org','doc_type','org_type', 'worker_focus', 'geography', 'date']
standards_data = standards[['title'] + columns_to_add].copy() # Grab columns as a copy

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
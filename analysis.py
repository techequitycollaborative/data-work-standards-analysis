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

# Load standards and framework data (for merging later)
standards = pd.read_csv('./data/standards.csv')
framework = pd.read_csv('./data/framework.csv')

# For our analysis, we want to look at:
# 1. Frequency of parameter queries per policy document 
# 2. Frequency of parameter queries overall
# 3. Distribution of parameter queries across different document variables
# 4. Adherence of each document to the framework / by document variables

#####  1. Frequency of parameter queries per policy document ##### 
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
plt.xlabel('Number of Documents with Parameter Present (Max 13)')
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
plt.xlabel('Number of Documents with Parameter Present (Max 13)')
plt.title('Parameter Frequency by Number of Documents')
plt.tight_layout()
plt.savefig('./plots/param_num_docs_barplot.png')
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
plt.savefig('./plots/param_by_doctype_heatmap.png')
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
plt.savefig('./plots/param_by_orgtype_heatmap.png')
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
plt.savefig('./plots/param_by_workerfocus_heatmap.png')
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
axes[1,2].set_xlabel('Document Length (corpus size)')
axes[1,2].set_ylabel('Proportion of Framework Covered')

plt.tight_layout()
plt.savefig('./plots/framework_adherence_analysis.png', dpi=300)
plt.show()
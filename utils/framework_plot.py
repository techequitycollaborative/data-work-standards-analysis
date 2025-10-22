"""
Title: framework_plot.py
Author: @dsherbini
Date: Oct 22, 2025
Description: Create table output for framework for readme file.
"""

# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap # for text wrapping in tables/plots
import json

# Load framework data
with open('../data/framework.json', 'r') as f:
    framework = json.load(f)

# Generate a table of framework parameters for readme
framework_params = pd.DataFrame(framework['framework'])
table_df = framework_params[['category', 'subcategory', 'parameter', 'definition']].copy()

# Optimized wrap text function
def wrap_text(text, width=50):
    return '\n'.join(textwrap.wrap(str(text), width=width, break_long_words=False))

# Apply wrapping with optimized widths for each column
table_df['category'] = table_df['category'].apply(lambda x: wrap_text(x, width=15))
table_df['subcategory'] = table_df['subcategory'].apply(lambda x: wrap_text(x, width=20))
table_df['parameter'] = table_df['parameter'].apply(lambda x: wrap_text(x, width=30))
table_df['definition'] = table_df['definition'].apply(lambda x: wrap_text(x, width=60))

# Calculate dynamic figure height based on content
max_lines_per_row = []
for idx, row in table_df.iterrows():
    max_lines = max(
        len(str(row['category']).split('\n')),
        len(str(row['subcategory']).split('\n')),
        len(str(row['parameter']).split('\n')),
        len(str(row['definition']).split('\n'))
    )
    max_lines_per_row.append(max_lines)

# Adjusted height calculation - smaller multiplier for tighter rows
total_height = sum(max_lines_per_row) * 0.16 + 1

# Create plot with optimized dimensions
fig, ax = plt.subplots(figsize=(18, total_height))
ax.axis('tight')
ax.axis('off')

# Prepare data for table
table_data = table_df.values.tolist()

# Create table with optimized column widths
param_table = ax.table(
    cellText=table_data,
    colLabels=['Category', 'Subcategory', 'Parameter', 'Definition'],
    cellLoc='left',
    loc='center',
    colWidths=[0.12, 0.15, 0.23, 0.50]
)

# Style the table
param_table.auto_set_font_size(False)
param_table.set_fontsize(10) 
param_table.scale(1, 1.3)

# Style header row
for i in range(4):
    cell = param_table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', size=11)  # Increased header font
    cell.set_height(0.05)

# Alternate row colors and adjust row heights
colors = ['#D9E1F2', '#FFFFFF']
for i in range(1, len(table_df) + 1):
    # Adjusted row height calculation - smaller multiplier
    row_height = max_lines_per_row[i-1] * 0.025  # Reduced from 0.04 to 0.025
    
    for j in range(4):
        cell = param_table[(i, j)]
        cell.set_facecolor(colors[i % 2])
        cell.set_edgecolor('gray')
        cell.set_height(row_height)
        
        # Adjust text properties for better readability
        cell.set_text_props(va='top', ha='left', size=10)  # Explicit size for cells

plt.tight_layout()
plt.savefig('../plots/framework.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nTable saved with {len(table_df)} parameters")
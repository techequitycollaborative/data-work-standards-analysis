"""
Title: text_extraction.py
Author: @dsherbini
Date: Oct 6, 2025

Extract text from .txt files containing data work standards and merge with existing standards dataframe.
"""
import pandas as pd
import os
import matplotlib.pyplot as plt

PATH = '/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/docs'
standards = pd.read_csv('/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/standards.csv')

####################### EXTRACT TEXT FROM TXT FILES #########################
def get_all_text(PATH):
    """
    Extract text from all TXT files in the specified folder.
    """
    all_text = []
    # Get all file names for policies in the folder
    for filename in os.listdir(PATH):
        # Only process .txt files
        if not filename.endswith('.txt'):
            continue
            
        file_path = os.path.join(PATH, filename)
        # Extract text from each TXT file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Append extracted text
            all_text.append({
                'filename': filename,
                'raw_text': text,
                'word_count': len(text.split()),
            })
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    text_df = pd.DataFrame(all_text)
    return text_df

# Extract text from all TXT files
text_df = get_all_text(PATH)

# Add text to original standards dataframe
text_df['filename'] = text_df['filename'].apply(lambda x: os.path.basename(x))  # Get base filenames to match with standards df
standards_merged = standards.merge(text_df, left_on='file_name', right_on='filename', how='left')
standards_merged = standards_merged.drop(columns=['filename'])  # Drop redundant column
print(standards_merged.columns)

# Display summary statistics of word counts to get a sense of document lengths
print(standards_merged['word_count'].describe())

# Plot word count by document
standards_merged_sorted = standards_merged.sort_values('word_count', ascending=True)
plt.figure(figsize=(12, max(8, len(standards_merged_sorted) * 0.3)))  # Dynamic height based on number of files
plt.barh(standards_merged_sorted['title'], standards_merged_sorted['word_count'], edgecolor='black', alpha=0.7)
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('File Name', fontsize=12)
plt.title('Word Count by Policy', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
# add word counts to each bar
for i, (filename, count) in enumerate(zip(standards_merged_sorted['title'], standards_merged_sorted['word_count'])):
    plt.text(count, i, f' {count}', va='center', fontsize=9)
plt.tight_layout()
plt.show()
plt.savefig('./plots/word_count_by_policy.png')

# Export updated standards df with extracted text to csv
standards_merged.to_csv('/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/standards_with_text.csv', index=False)
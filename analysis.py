"""
Title: analysis.py
Author: @dsherbini
Date: Oct 6, 2025

Analysis of standards/policies for ethical data work in the AI supply chain.
"""

import pandas as pd
from utils.pdf_parser import extract_pdf_text
import os

PATH = '/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/docs'
standards = pd.read_csv('/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/standards.csv')

####################### EXTRACT TEXT FROM PDFs #########################
def get_all_text(PATH):
    """
    Extract text from all PDF files in the specified folder.
    """
    all_text = []

    # Get all file names for policies in the folder
    for filename in os.listdir(PATH):
        file_path = os.path.join(PATH, filename)
    
        # Extract text from each PDF file
        try:
            text = extract_pdf_text(file_path)

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

# Extract text from all PDF files
text_df = get_all_text(PATH)

# Display summary statistics of word counts to get a sense of document lengths
print(text_df['word_count'].describe())

# Add text to original standards dataframe
text_df['filename'] = text_df['filename'].apply(lambda x: os.path.basename(x)) # Get base filenames to match with standards df
standards_merged = standards.merge(text_df, left_on='file_name', right_on='filename', how='left')
standards_merged = standards_merged.drop(columns=['filename']) # Drop redundant column
print(standards_merged.columns)

# Export updated standards df with extracted text to csv
standards_merged.to_csv('/Users/danyasherbini/Documents/GitHub/data-work-standards-analysis/data/standards_with_text.csv', index=False)
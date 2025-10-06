"""
Title: pdf_parser.py
Author: @dsherbini
Date: Oct 6, 2025

Function for parsing PDF files and extracting text content.
"""

import pandas as pd
import pdfplumber
import os

def extract_pdf_text(file_path):
    """
    Extracts text from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with pdfplumber.open(file_path) as pdf:
        text = ' '.join(page.extract_text() for page in pdf.pages)
    
    return text




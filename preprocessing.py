# ============================================
# preprocessing.py
# This file cleans the text data before we use it.
# Steps: lowercase -> remove punctuation -> split into words
# ============================================

import re       # re = regular expressions, used to find/remove patterns in text
import pandas as pd   # pandas helps us work with tables (CSV files)
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split()


def preprocess_data(df):
    """
    Clean the entire dataset (DataFrame).
    - Remove rows with missing values
    - Clean each review
    - Split each review into words (tokens)
    Returns the cleaned DataFrame.
    """
    # Make a copy so we don't change the original data
    df = df.copy()

    # Remove rows where 'review' or 'sentiment' is empty/missing
    df = df.dropna(subset=["review", "sentiment"])

    # Remove rows where review is just whitespace
    df = df[df["review"].str.strip() != ""]

    # Clean every review using our clean_text function
    df["review"] = df["review"].apply(clean_text)

    # Tokenize every review (split into word lists)
    df["tokens"] = df["review"].apply(tokenize)

    # Reset the index (row numbers) starting from 0
    df = df.reset_index(drop=True)

    return df

# ============================================
# preprocessing.py
# This file cleans the text data before we use it.
# Steps: lowercase -> remove punctuation -> split into words
# ============================================

import re       # re = regular expressions, used to find/remove patterns in text
import pandas as pd   # pandas helps us work with tables (CSV files)


def clean_text(text):
    """
    This function takes a single review string and cleans it.
    Example: "Great Product!!!" -> "great product"
    """
    # Step 1: If the input is not a string (maybe NaN), return empty string
    if not isinstance(text, str):
        return ""

    # Step 2: Convert everything to lowercase
    # "GREAT Product" -> "great product"
    text = text.lower()

    # Step 3: Remove anything that is NOT a letter or space
    # This removes numbers, punctuation, symbols etc.
    # [^a-z\s] means "anything that is not a-z or whitespace"
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 4: Remove extra spaces (multiple spaces become one)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    """
    Split a sentence into a list of words.
    Example: "good product" -> ["good", "product"]
    """
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

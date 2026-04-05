# ============================================
# feature_extraction.py
# This file converts text into numbers so the
# Perceptron can understand it.
#
# We use "Bag of Words" (BoW) method:
#   - First, make a list of ALL unique words (vocabulary)
#   - Then, for each review, count how many times each word appears
#   - This gives us a number vector for each review
# ============================================

import numpy as np   # numpy is used for math operations and arrays


def build_vocabulary(list_of_token_lists):
    """
    Build a vocabulary dictionary from training data.

    Input:  [["good", "product"], ["bad", "product"], ["good", "quality"]]
    Output: {"good": 0, "product": 1, "bad": 2, "quality": 3}

    Each unique word gets a unique number (index).
    """
    vocabulary = {}     # empty dictionary to store words
    index = 0           # counter for word index

    # Go through each review's word list
    for tokens in list_of_token_lists:
        # Go through each word in the review
        for word in tokens:
            # If this word is new (not seen before), add it
            if word not in vocabulary:
                vocabulary[word] = index
                index = index + 1

    return vocabulary


def text_to_vector(tokens, vocabulary):
    """
    Convert one review (list of words) into a number vector.

    Example:
      vocabulary = {"good": 0, "bad": 1, "product": 2}
      tokens = ["good", "good", "product"]
      output = [2, 0, 1]  (good appears 2 times, bad 0 times, product 1 time)
    """
    # Create a vector of zeros, size = number of words in vocabulary
    vector = np.zeros(len(vocabulary))

    # For each word in the review, increase its count in the vector
    for word in tokens:
        if word in vocabulary:
            position = vocabulary[word]
            vector[position] = vector[position] + 1

    return vector


def extract_features(list_of_token_lists, vocabulary):
    """
    Convert ALL reviews into a feature matrix (2D array).

    Each row = one review's vector
    Each column = one word from vocabulary

    Output shape: (number_of_reviews, vocabulary_size)
    """
    num_reviews = len(list_of_token_lists)
    vocab_size = len(vocabulary)

    # Create a big matrix of zeros
    X = np.zeros((num_reviews, vocab_size))

    # Fill each row with the vector for that review
    for i in range(num_reviews):
        X[i] = text_to_vector(list_of_token_lists[i], vocabulary)

    return X


def extract_labels(df):
    """
    Get the sentiment column as a numpy array.
    This is our 'y' (target/output) for training.
    """
    y = df["sentiment"].values.astype(int)
    return y

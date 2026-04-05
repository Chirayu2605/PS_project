# ============================================
# train.py
# This is the training pipeline.
# It connects all the other files together:
#   1. Load the CSV dataset
#   2. Clean the text
#   3. Build vocabulary
#   4. Convert text to numbers (Bag of Words)
#   5. Split into training and testing sets
#   6. Train the Perceptron
#   7. Evaluate the model
#   8. Show results
# ============================================

import numpy as np
import pandas as pd

# Import our own modules
from preprocessing import preprocess_data
from feature_extraction import build_vocabulary, extract_features, extract_labels
from perceptron import Perceptron
from evaluation import (compute_accuracy, confusion_matrix,
                        print_confusion_matrix, show_sample_predictions,
                        plot_accuracy)


def load_dataset(filepath):
    """
    Load the CSV file and shuffle the rows randomly.
    Shuffling is important so the model doesn't learn the order of data.
    """
    df = pd.read_csv(filepath)

    # Shuffle all rows randomly (random_state=42 makes it reproducible)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def split_data(X, y, reviews, test_size=0.2):
    """
    Split data into training set (80%) and testing set (20%).
    We do this manually without using sklearn.

    Why split?
    - We train on 80% of data
    - We test on the remaining 20% to see how well the model works
      on data it has never seen before
    """
    total = len(y)
    split_point = int(total * (1 - test_size))   # 80% of total

    # Training data (first 80%)
    X_train = X[:split_point]
    y_train = y[:split_point]
    reviews_train = reviews[:split_point]

    # Testing data (last 20%)
    X_test = X[split_point:]
    y_test = y[split_point:]
    reviews_test = reviews[split_point:]

    return X_train, X_test, y_train, y_test, reviews_train, reviews_test


def run_pipeline(csv_path="dataset.csv", learning_rate=0.01, n_epochs=15):
    """
    This function runs the ENTIRE project step by step.
    """

    # ---- STEP 1: Load the dataset ----
    print("=" * 60)
    print(" BINARY SENTIMENT ANALYSIS -- PERCEPTRON (FROM SCRATCH)")
    print("=" * 60)
    print("\n[1] Loading dataset ...")
    df = load_dataset(csv_path)
    print(f"    Loaded {len(df)} rows.")

    # ---- STEP 2: Preprocess (clean) the text ----
    print("[2] Preprocessing text ...")
    df = preprocess_data(df)
    print(f"    After cleaning: {len(df)} rows.")

    # ---- STEP 3: Prepare data for splitting ----
    reviews_list = df["review"].tolist()       # list of cleaned review strings
    token_lists = df["tokens"].tolist()         # list of word lists
    y = extract_labels(df)                      # sentiment labels as numpy array

    # Split into 80% training and 20% testing
    total = len(y)
    split_point = int(total * 0.8)

    train_tokens = token_lists[:split_point]
    test_tokens = token_lists[split_point:]

    # ---- STEP 4: Build vocabulary (ONLY from training data) ----
    # Important: We only use training data to build vocabulary
    # because in real life, we don't know the test data beforehand
    print("[3] Building vocabulary from training data ...")
    vocab = build_vocabulary(train_tokens)
    print(f"    Vocabulary size: {len(vocab)} unique words.")

    # ---- STEP 5: Convert text to numerical vectors (Bag of Words) ----
    print("[4] Extracting Bag-of-Words features ...")
    X_train = extract_features(train_tokens, vocab)
    X_test = extract_features(test_tokens, vocab)

    y_train = y[:split_point]
    y_test = y[split_point:]

    reviews_train = reviews_list[:split_point]
    reviews_test = reviews_list[split_point:]

    print(f"    Training set : {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"    Test set     : {X_test.shape[0]} samples")

    # ---- STEP 6: Create and train the Perceptron ----
    print(f"\n[5] Training Perceptron (lr={learning_rate}, epochs={n_epochs}) ...\n")
    model = Perceptron(learning_rate=learning_rate, n_epochs=n_epochs)
    epoch_accuracies = model.fit(X_train, y_train)

    # ---- STEP 7: Test the model on unseen data ----
    print("\n[6] Evaluating on test set ...")
    y_pred = model.predict(X_test)

    # ---- STEP 8: Calculate and show results ----
    test_accuracy = compute_accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n    * Test Accuracy: {test_accuracy:.4f}  ({test_accuracy*100:.2f}%)")
    print_confusion_matrix(cm)

    # Show some example predictions
    show_sample_predictions(reviews_test, y_test, y_pred, n=10)

    # Save accuracy plot
    plot_accuracy(epoch_accuracies)

    # Return the model and vocab so main.py can use them
    return model, vocab, test_accuracy, cm

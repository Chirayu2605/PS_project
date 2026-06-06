# ============================================
# perceptron.py
# This is the CORE of our project.
#
# The Perceptron is the simplest machine learning model.
# It works like a single brain cell (neuron).
#
# How it works:
#   1. Multiply each input by a weight and add them up
#      z = w1*x1 + w2*x2 + ... + wn*xn + bias
#   2. If z >= 0, predict 1 (Positive)
#      If z < 0,  predict 0 (Negative)
#   3. If prediction is wrong, adjust the weights
#
# Weight Update Rule:
#   error = actual_label - predicted_label
#   new_weight = old_weight + learning_rate * error * input
#   new_bias   = old_bias   + learning_rate * error
# ============================================

import numpy as np


class Perceptron:
    """
    A simple Perceptron classifier.

    Parameters:
        learning_rate : how fast the model learns (default 0.01)
        n_epochs      : how many times to go through the entire dataset
    """

    def __init__(self, learning_rate=0.01, n_epochs=15):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None         # will be set during training
        self.bias = 0.0             # starts at zero
        self.accuracy_history = []  # stores accuracy of each epoch

    def activation(self, z):
        """
        Step function (activation function).
        If z > 0, return 1 (Positive)
        If z <= 0, return 0 (Negative)

        Note: When z is exactly 0 (e.g. all unknown words),
        we default to Negative to avoid false positives.
        """
        if z > 0:
            return 1
        else:
            return 0

    def predict_one(self, x):
        """
        Predict the class for ONE input sample.
        x is a 1D numpy array (one review's feature vector).

        Steps:
          1. Calculate z = sum of (weights * inputs) + bias
          2. Pass z through activation function
        """
        # np.dot does the multiplication and addition for us
        # z = w1*x1 + w2*x2 + ... + wn*xn + bias
        z = np.dot(self.weights, x) + self.bias

        # Apply activation function
        prediction = self.activation(z)

        return prediction

    def fit(self, X, y):
        """
        Train the perceptron model.

        X : feature matrix (rows = samples, columns = features)
        y : labels array (0 or 1 for each sample)

        Returns: list of accuracy values for each epoch
        """
        num_samples = X.shape[0]     # number of training examples
        num_features = X.shape[1]    # number of features (vocabulary size)

        # Step 1: Initialize all weights to zero
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        self.accuracy_history = []

        # Step 2: Train for multiple epochs
        for epoch in range(1, self.n_epochs + 1):
            correct_count = 0   # count correct predictions in this epoch

            # Go through each training sample one by one
            for i in range(num_samples):
                # Get prediction for this sample
                y_pred = self.predict_one(X[i])

                # Calculate error
                error = y[i] - y_pred
                # error =  0  means correct prediction (no update needed)
                # error =  1  means we predicted 0 but actual was 1
                # error = -1  means we predicted 1 but actual was 0

                # Update weights and bias (only happens when error != 0)
                self.weights = self.weights + self.learning_rate * error * X[i]
                self.bias = self.bias + self.learning_rate * error

                # Count correct predictions
                if y_pred == y[i]:
                    correct_count = correct_count + 1

            # Calculate accuracy for this epoch
            accuracy = correct_count / num_samples
            self.accuracy_history.append(accuracy)
            print(f"  Epoch {epoch:>2}/{self.n_epochs}  --  Training accuracy: {accuracy:.4f}")

        return self.accuracy_history

    def predict(self, X):
        """
        Predict classes for MULTIPLE samples.
        X is a 2D numpy array (rows = samples).
        Returns a numpy array of predictions (0s and 1s).
        """
        predictions = []
        for i in range(X.shape[0]):
            pred = self.predict_one(X[i])
            predictions.append(pred)

        return np.array(predictions)

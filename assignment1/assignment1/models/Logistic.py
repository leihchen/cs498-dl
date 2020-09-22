"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N = X_train.shape[0]  # Number of pictures
        D = X_train.shape[1]  # Picture dim
        self.w = np.random.rand(D)  # init all class w as random
        for e in range(self.epochs):
            for i, data_point in enumerate(X_train):
                y_prime = 1 if y_train[i] == 1 else -1
                self.w += self.lr * self.sigmoid(- y_prime * np.inner(self.w, data_point)) * y_prime * data_point
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        retval = np.zeros(len(X_test)).astype(int)
        for i, data_point in enumerate(X_test):
            prediction = np.inner(data_point, self.w)
            retval[i] = prediction > 0.5
        print(retval)
        return retval

"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = [[] for _ in range(n_class)]  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in Lecture 3.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N = X_train.shape[0]  # Number of pictures
        D = X_train.shape[1]  # Picture dim
        for i in range(len(self.w)):
            self.w[i] = np.random.rand(D)    # init all class w as random

        for e in range(self.epochs):
            for i, data_point in enumerate(X_train):
                result = self.predict([data_point])[0]
                if result != y_train[i]:
                    self.w[result] -= self.lr * data_point
                    self.w[y_train[i]] += self.lr * data_point
                # if np.sign(np.inner(self.w, data_point)) != y_train[i]:
                #     self.w += self.lr * y_train[i] * data_point
            # print('Trained ', e, ' epochs: train_acc=', np.sum(self.predict(X_train) == y_train) / len(y_train) * 100, '%')
            # self.lr /= 2
        # print('finish training')

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
            prediction = [np.inner(data_point, self.w[n]) for n in range(self.n_class)]  # len n
            retval[i] = int(np.argsort(prediction)[::-1][0])
        return retval

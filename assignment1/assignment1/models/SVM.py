"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, batch_num):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = [[] for _ in range(n_class)]  # TODO: change this
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.batch_size = batch_num

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        grad_w_yi = np.zeros(self.w.shape)
        grad_w_yi += self.reg_const * abs(self.w)
        for i, data_point in enumerate(X_train):
            result = [np.inner(data_point, self.w[n]) for n in range(self.n_class)]
            gt_label = y_train[i]
            for c in range(self.n_class):
                if c != gt_label and result[gt_label] - result[c] < 1:
                    grad_w_yi[gt_label] -= data_point / len(data_point)  # BUG: sign
                    grad_w_yi[c] += data_point / len(data_point)
        return grad_w_yi

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = X_train.shape
        for i in range(len(self.w)):
            self.w[i] = np.random.rand(D) * 0.0001    # init all class w as random
        self.w = np.array(self.w)
        # implements mini-batch gradient descent
        n_batch = int(N / self.batch_size)
        # random shuffle X_train, y_train
        n_shuffle = np.random.permutation(N)
        X_train_cp, y_train_cp = X_train.copy()[n_shuffle], y_train.copy()[n_shuffle]
        for e in range(self.epochs):
            for n in range(n_batch):
                start = self.batch_size * n
                end = self.batch_size * (n + 1)
                grad = self.calc_gradient(X_train_cp[start:end], y_train_cp[start:end])
                self.w -= self.alpha * grad
            # print('Trained ', e, ' epochs: train_acc=', np.sum(self.predict(X_train) == y_train) / len(y_train) * 100, '%')
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
            prediction = [np.inner(data_point, self.w[n]) for n in range(self.n_class)]  # len n
            retval[i] = int(np.argsort(prediction)[::-1][0])
        return retval
# Reference: Lecture 5, https://piazza.com/class/kdyxzd9ldz23vn?cid=66

"""Neural network model."""

from typing import Sequence

import numpy as np


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, 0) ndarray
    Returns: scalar
    """
    # https://cs231n.github.io/neural-networks-case-study/
    num_examples = len(predictions)
    correct_logprobs = -np.log(predictions[range(num_examples), targets] + epsilon)
    data_loss = np.sum(correct_logprobs) / num_examples
    return data_loss


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
        self.adam_m = {}
        self.adam_v = {}
        for k in self.params:
            print('init adam m, v for', k)
            self.adam_m[k] = np.zeros(self.params[k].shape)
            self.adam_v[k] = np.zeros(self.params[k].shape)

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        # TODO: implement me
        return np.dot(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        # https://stackoverflow.com/questions/42599498/numercially-stable-softmax
        z = X - np.max(X, axis=-1, keepdims=True)
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        return numerator / denominator

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.
        prev_layer = X
        for l in range(1, self.num_layers + 1):
            param_W = self.params["W" + str(l)]
            param_b = self.params["b" + str(l)]
            self.outputs[l] = self.linear(param_W, prev_layer, param_b)  # @ fixme
            prev_layer = self.relu(self.outputs[l])
            # print("prev_layer", prev_layer)
        return self.softmax(self.outputs[self.num_layers])

    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0, do_sgd_update: bool = True
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength
            do_sgd_update: don't update weights when running debug or Adam
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        for key in self.params:
            self.gradients[key] = np.zeros(self.params[key].shape)
        N = len(X)
        C = self.output_size
        final_output = self.softmax(self.outputs[self.num_layers])
        gt = np.zeros((N, C)).astype(float)  # one-hot ground truth y
        for i in range(len(y)):
            gt[i, y[i]] = 1
        loss = cross_entropy(final_output, y)  # cross-entropy loss
        loss += np.mean([reg * np.sum(self.params["W" + str(i)] ** 2) for i in range(1, self.num_layers + 1)])  # W-reg
        # TODO: implement me. You'll want to store the gradient of each layer
        # in self.gradients if you want to be able to debug your gradients
        # later. You can use the same keys as self.params. You can add
        # functions like self.linear_grad, self.relu_grad, and
        # self.softmax_grad if it helps organize your code.
        # https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/gradient-notes.pdf
        if self.num_layers == 2:
            for nn in range(N):
                delta1 = (final_output[nn] - gt[nn])  # delta @ layer -1
                h = self.relu(self.outputs[1][nn])
                delta2 = self.params["W" + str(2)] @ delta1 * np.sign(h)
                dW2 = np.outer(h, delta1)
                db2 = delta1
                dW = np.outer(X[nn], delta2)
                db = delta2
                self.gradients["W" + str(1)] += dW / N
                self.gradients["W" + str(2)] += dW2 / N
                self.gradients["b" + str(1)] += db / N
                self.gradients["b" + str(2)] += db2 / N
            dW = self.gradients["W" + str(1)] + reg * self.params["W" + str(1)]
            dW2 = self.gradients["W" + str(2)] + reg * self.params["W" + str(2)]
            db = self.gradients["b" + str(1)] + reg * self.params["b" + str(1)]
            db2 = self.gradients["b" + str(2)] + reg * self.params["b" + str(2)]
            if do_sgd_update:
                self.params["W" + str(1)] -= lr * dW
                self.params["W" + str(2)] -= lr * dW2
                self.params["b" + str(1)] -= lr * db
                self.params["b" + str(2)] -= lr * db2
        elif self.num_layers == 3:
            for nn in range(N):
                delta1 = (final_output[nn] - gt[nn])  # delta @ layer -1
                h = self.relu(self.outputs[2][nn])
                delta2 = self.params["W" + str(3)] @ delta1 * np.sign(h)  # delta @ layer -2
                dW3 = np.outer(h, delta1)
                db3 = delta1
                dW2 = np.outer(self.outputs[1][nn], delta2)
                db2 = delta2

                h1 = self.relu(self.outputs[1][nn])
                delta3 = self.params["W" + str(2)] @ delta2 * np.sign(h1)  # delta @ layer -3
                dW1 = np.outer(X[nn], delta3)
                db1 = delta3
                self.gradients["W" + str(1)] += dW1 / N
                self.gradients["W" + str(2)] += dW2 / N
                self.gradients["W" + str(3)] += dW3 / N
                self.gradients["b" + str(1)] += db1 / N
                self.gradients["b" + str(2)] += db2 / N
                self.gradients["b" + str(3)] += db3 / N
            dW = self.gradients["W" + str(1)] + reg * self.params["W" + str(1)]
            dW2 = self.gradients["W" + str(2)] + reg * self.params["W" + str(2)]
            dW3 = self.gradients["W" + str(3)] + reg * self.params["W" + str(3)]
            db = self.gradients["b" + str(1)] + reg * self.params["b" + str(1)]
            db2 = self.gradients["b" + str(2)] + reg * self.params["b" + str(2)]
            db3 = self.gradients["b" + str(3)] + reg * self.params["b" + str(3)]
            if do_sgd_update:
                self.params["W" + str(1)] -= lr * dW
                self.params["W" + str(2)] -= lr * dW2
                self.params["W" + str(3)] -= lr * dW3
                self.params["b" + str(1)] -= lr * db
                self.params["b" + str(2)] -= lr * db2
                self.params["b" + str(3)] -= lr * db3
        else:
            print("Only 2 and 3 layer network supported")
            return NotImplemented
        return loss

    def backward_adam(
        self, X: np.ndarray, y: np.ndarray, lr: float = 1e-3, beta1: float = 0.9, beta2 :float = 0.999, reg: float = 0.0, epsilon=1e-8
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            beta1: beta_1 in Adam
            beta2: beta_21 in Adam
            reg: Regularization strength
            epsilon: numerical small value to avoid division by zero

        Returns:
            Total loss for this batch of training samples
        """
        loss = self.backward(X, y, lr, reg, False)
        for k in self.params:  # for each params:
            self.adam_m[k] = beta1 * self.adam_m[k] + (1 - beta1) * self.gradients[k]
            self.adam_v[k] = beta2 * self.adam_v[k] + (1 - beta2) * (self.gradients[k]) ** 2
            m_hat = self.adam_m[k] / (1 - beta1)
            v_hat = self.adam_v[k] / (1 - beta2)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        return loss

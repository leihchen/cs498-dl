"""Neural network model."""

from typing import Sequence

import numpy as np

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    # predictions = np.clip(predictions, epsilon, 1. - epsilon)
    # N = predictions.shape[0]
    # ce = -np.sum(targets*np.log(predictions))/N
    # print(ce)
    num_examples = predictions.shape[0]
    correct_logprobs = -np.log(predictions[:, targets])
    ce = np.sum(correct_logprobs) / num_examples
    print(ce)
    return ce

class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are the scores for
    each class."""

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
        # print("W", W.shape)
        # print("X", X.shape)
        return np.dot(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        # _X = X.copy()
        # _X[X < 0] = 0
        # print(np.min(_X))
        return np.maximum(0, X)

    # def grad_relu(self, X: np.ndarray) -> np.ndarray:
    #     _X = X.copy()
    #     _X[X <= 0] = 0
    #     _X[X > 0] = 1
    #     return _X

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        exp_scores = np.exp(X)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

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
        activation = 'relu'
        # prev_layer = X
        # for l in range(1, self.num_layers + 1):
        #     param_W = self.params["W" + str(l)]
        #     param_b = self.params["b" + str(l)]
        #     self.outputs[l] = self.linear(param_W, prev_layer, param_b)  # @ fixme
        #     prev_layer = self.relu(self.outputs[l])
        #     # print("prev_layer", prev_layer)
        # return self.softmax(self.outputs[self.num_layers])
        self.outputs[1] = self.linear(self.params["W" + str(1)], X, self.params["b" + str(1)])
        act1 = self.relu(self.outputs[1])
        self.outputs[2] = self.linear(self.params["W" + str(2)], act1, self.params["b" + str(2)])
        act2 = self.softmax(self.outputs[2])
        return act2

    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        final_output = self.forward(X)
        # print("y hat=", final_output)
        self.gradients = {}
        for key in self.params:
            self.gradients[key] = np.zeros(self.params[key].shape)
        N = len(X)
        C = self.output_size
        gt = np.zeros((N, C)).astype(float)  # one-hot ground truth y
        for i in range(len(y)):
            gt[i, y[i]] = 1
        y_hat = np.zeros((N, C)).astype(float)
        for i in range(len(y)):
            y_hat[i, np.argmax(final_output[i])] = 1
        loss = cross_entropy(final_output, y)
        # TODO: implement me. You'll want to store the gradient of each layer
        # in self.gradients if you want to be able to debug your gradients
        # later. You can use the same keys as self.params. You can add
        # functions like self.linear_grad, self.relu_grad, and
        # self.softmax_grad if it helps organize your code.


        # prev_out = self.outputs[l - 1]
        # this_W = self.params["W" + str(l)]
        # this_b = self.params["b" + str(l)]
        assert self.num_layers == 2
        dscores = (final_output - gt) * (1 / len(X))
        dscores[:, y] -= 1
        dscores /= len(X)
        # dscores = np.multiply(dscores, final_output)
        hidden_layer = self.relu(self.outputs[1])
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)

        dhidden = np.dot(dscores, self.params["W" + str(2)].T)
        dhidden[hidden_layer <= 0] = 0
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)
        self.gradients["W" + str(1)] = dW
        self.gradients["W" + str(2)] = dW2
        # self.gradients["b" + str(1)] = db
        # self.gradients["b" + str(2)] = db2
        # for nn in range(N):
        #     delta1 = (final_output[nn] - gt[nn])  # delta @ layer -1
        #     h = self.relu(self.outputs[1][nn])
        #     # print(h)
        #     delta2 = self.params["W" + str(2)] @ delta1 * np.sign(h)  # # delta @ layer -2 @fixme: h OR self.outputs[1][nn]
        #     dldW2 = np.outer(h, delta1)
        #     dldb2 = delta1
        #     dldW1 = np.outer(X[nn], delta2)
        #     dldb1 = delta2
        #     self.gradients["W" + str(1)] += dldW1 / N
        #     self.gradients["W" + str(2)] += dldW2 / N
        #     self.gradients["b" + str(1)] += dldb1 / N
        #     self.gradients["b" + str(2)] += dldb2 / N
        # self.params["W" + str(1)] -= lr * self.gradients["W" + str(1)]#+ np.abs(self.params["W" + str(1)]) * reg)
        # self.params["W" + str(2)] -= lr * self.gradients["W" + str(2)]#+ np.abs(self.params["W" + str(2)]) * reg)
        # self.params["b" + str(1)] -= lr * self.gradients["b" + str(1)]#+ np.abs(self.params["b" + str(1)]) * reg)
        # self.params["b" + str(2)] -= lr * self.gradients["b" + str(2)]#+ np.abs(self.params["b" + str(2)]) * reg)
        # print(self.gradients)
        return loss

    # x = input
    # z = W
    # x + b1
    # h = ReLU(z)
    # θ = U
    # h + b2
    # yˆ = softmax(θ)
    # J = C
    # E(y, yˆ)
    # def back_prop(self, y_hat, y, U, z):
    #     delta1 = y_hat - y
    #     delta2 = (delta1.T @ U) * np.sign(z)
    #

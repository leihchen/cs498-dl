#!/usr/bin/env python
# coding: utf-8

# # Implement a Neural Network
# 
# This notebook contains useful information and testing code to help you to develop a neural network by implementing the forward pass and backpropagation algorithm in the `models/neural_net.py` file. 

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

from models.neural_net import NeuralNetwork

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots

# For auto-reloading external modules
# See http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

def rel_error(x, y):
    """Returns relative error"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# You will implement your network in the class `NeuralNetwork` inside the file `models/neural_net.py` to represent instances of the network. The network parameters are stored in the instance variable `self.params` where keys are string parameter names and values are numpy arrays.
# 
# The cell below initializes a toy dataset and corresponding model which will allow you to check your forward and backward pass by using a numeric gradient check.

# In[ ]:


# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5


def init_toy_model(num_layers):
    np.random.seed(0)
    hidden_sizes = [hidden_size] * (num_layers - 1)
    return NeuralNetwork(input_size, hidden_sizes, num_classes, num_layers)

def init_toy_data():
    np.random.seed(0)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.random.randint(num_classes, size=num_inputs)
    return X, y


# # Implement forward and backward pass
# 
# The first thing you will do is implement the forward pass of your neural network along with the loss calculation. The forward pass should be implemented in the `forward` function. You can use helper functions like `linear`, `relu`, and `softmax` to help organize your code.
# 
# Next, you will implement the backward pass using the backpropagation algorithm. Backpropagation will compute the gradient of the loss with respect to the model parameters `W1`, `b1`, ... etc. Use a softmax fuction with cross entropy loss for loss calcuation. Fill in the code blocks in `NeuralNetwork.backward`. 

# # Gradient  check
# 
#  If you have implemented your forward pass through the network correctly, you can use the following cell to debug your backward pass with a numeric gradient check:

# In[ ]:


from utils.gradient_check import eval_numerical_gradient

X, y = init_toy_data()
learning_rate = 1e-1
# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be around 1e-8 or less for each of the parameters 
# W1, b1, ... in your network.

# for num in [2]:
#     net = init_toy_model(num)
#     loss = net.backward(X, y, learning_rate, reg=0.05)
#
#     # These should all be less than 1e-8 or so
#     for param_name in net.gradients:
#         net_params = net.params.copy()
#         f = lambda W: net.backward(X, y, learning_rate, reg=0.05)
#         param_grad_num = eval_numerical_gradient(f, net_params[param_name], verbose=False)

for num in [2]:
    net = init_toy_model(num)
    loss = net.backward(X, y, learning_rate, reg=0.05)

    # These should all be less than 1e-8 or so
    for param_name in net.gradients:
        f = lambda W: net.backward(X, y, learning_rate, reg=0.05)
        param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
        print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, net.gradients[param_name])))
        assert param_grad_num.shape == net.gradients[param_name].shape
        print(param_name, ' param_grad_num: ')
        print(param_grad_num)
        print(param_name, ' net.gradients: ')
        print(net.gradients[param_name])
        print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, net.gradients[param_name])))


# # Train the network
# To train the network we will use stochastic gradient descent (SGD), similar to the SVM and Softmax classifiers you trained. This should be similar to the training procedure you used for the SVM and Softmax classifiers.
# 
# Once you have implemented SGD, run the code below to train a two-layer network on toy data. You should achieve a training loss less than 0.2 using a two-layer network with relu activation.

# In[ ]:


# Hyperparameters
epochs = 100
batch_size = 1
learning_rate = 1e-1
learning_rate_decay = 0.95
regularization = 5e-6

# Initialize a new neural network model
net = init_toy_model(2)

# Variables to store performance for each epoch
train_loss = np.zeros(epochs)
train_accuracy = np.zeros(epochs)

# For each epoch...
for epoch in range(epochs):        
    # Training
    # Run the forward pass of the model to get a prediction and compute the accuracy
    # Run the backward pass of the model to update the weights and compute the loss
    train_loss[epoch] = 0
    train_accuracy[epoch] = 0


# In[ ]:


# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(train_loss)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(train_accuracy)
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.show()


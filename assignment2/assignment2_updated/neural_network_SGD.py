#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # CS498DL Assignment 2

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from kaggle_submission import output_submission_csv
from models.neural_net import NeuralNetwork
from utils.data_process import get_CIFAR10_data
import pickle
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots

# For auto-reloading external modules
# See http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# ## Loading CIFAR-10
# Now that you have implemented a neural network that passes gradient checks and works on toy data, you will test your network on the CIFAR-10 dataset.

# In[2]:


# You can change these numbers for experimentation
# For submission be sure they are set to the default values
TRAIN_IMAGES = 49000
VAL_IMAGES = 1000
TEST_IMAGES = 10000

data = get_CIFAR10_data(TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES)
X_train, y_train = data['X_train'], data['y_train']
print((X_train.max()))
print((X_train.min()))
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']


# ## Train using SGD
# To train our network we will use SGD. In addition, we will adjust the learning rate with an exponential learning rate schedule as optimization proceeds; after each epoch, we will reduce the learning rate by multiplying it by a decay rate.
# 
# You can try different numbers of layers and other hyperparameters on the CIFAR-10 dataset below.

# In[ ]:


# Hyperparameters

learning_rate_l = [1e-3, 1e-2]
learning_rate_decay_l = [0.95]
regularization_l = [0.1]
hidden_size_l = [20, 60]





# Initialize a new neural network model
d = {}
for learning_rate in learning_rate_decay_l:
    for learning_rate_decay in learning_rate_decay_l:
        for regularization in regularization_l:
            for hidden_size in hidden_size_l:
                input_size = 32 * 32 * 3
                num_layers = 2
                hidden_sizes = [hidden_size] * (num_layers - 1)
                num_classes = 10
                epochs = 100
                batch_size = 200
                net = NeuralNetwork(input_size, hidden_sizes, num_classes, num_layers)

                # Variables to store performance for each epoch
                train_loss = np.zeros(epochs)
                train_accuracy = np.zeros(epochs)
                val_accuracy = np.zeros(epochs)

                # For each epoch...
                for epoch in range(epochs):
                    print('epoch:', epoch)
                    if epoch > 10 and (np.abs(np.diff(train_accuracy[:-10] / len(X_train))) < 1e-4).all():
                        print("Early stop at epoch = ", epoch)
                        break
                    # Shuffle the dataset
                    n_shuffle = np.random.permutation(len(X_train))
                    X_train_cp, y_train_cp = X_train.copy()[n_shuffle], y_train.copy()[n_shuffle]
                    # Training
                    # For each mini-batch...
                    for batch in range(TRAIN_IMAGES // batch_size):
                        # Create a mini-batch of training data and labels
                        start = batch_size * batch
                        end = batch_size * (batch + 1)
                        X_batch = X_train_cp[start:end]
                        y_batch = y_train_cp[start:end]

                        # Run the forward pass of the model to get a prediction and compute the accuracy
                        scores = net.forward(X_batch)
                        predicted_class = np.argmax(scores, axis=1)
                        # Run the backward pass of the model to update the weights and compute the loss
                        loss = net.backward(X_batch, y_batch, learning_rate, regularization)
                        train_loss[epoch] += loss
                        train_accuracy[epoch] += np.sum(predicted_class == y_batch)
                    learning_rate *= learning_rate_decay
                    print("train acc = ", train_accuracy[epoch] / len(X_train))
                    # Validation
                    # No need to run the backward pass here, just run the forward pass to compute accuracy
                    val_accuracy[epoch] += np.mean(np.argmax(net.forward(X_val, const=True), axis=1) == y_val)
                d[(learning_rate, learning_rate_decay, regularization)] = [train_loss, train_accuracy, val_accuracy]
with open("SGDout_layer"+str(num_layers)+".pkl", "wb+") as f:
    pickle.dump(d, f)



# ## Train using Adam
# Next we will train the same model using the Adam optimizer. You should take the above code for SGD and modify it to use Adam instead. For implementation details, see the lecture slides. The original paper that introduced Adam is also a good reference, and contains suggestions for default values: https://arxiv.org/pdf/1412.6980.pdf

# In[ ]:


# TODO: implement me


# ## Graph loss and train/val accuracies
# 
# Examining the loss graph along with the train and val accuracy graphs should help you gain some intuition for the hyperparameters you should try in the hyperparameter tuning below. It should also help with debugging any issues you might have with your network.

# In[ ]:


# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(train_loss)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(train_accuracy, label='train')
plt.plot(val_accuracy, label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.show()


# ## Hyperparameter tuning
# 
# Once you have successfully trained a network you can tune your hyparameters to increase your accuracy.
# 
# Based on the graphs of the loss function above you should be able to develop some intuition about what hyperparameter adjustments may be necessary. A very noisy loss implies that the learning rate might be too high, while a linearly decreasing loss would suggest that the learning rate may be too low. A large gap between training and validation accuracy would suggest overfitting due to large model without much regularization. No gap between training and validation accuracy would indicate low model capacity. 
# 
# You will compare networks of two and three layers using the different optimization methods you implemented. 
# 
# The different hyperparameters you can experiment with are:
# - **Batch size**: We recommend you leave this at 200 initially which is the batch size we used. 
# - **Number of iterations**: You can gain an intuition for how many iterations to run by checking when the validation accuracy plateaus in your train/val accuracy graph.
# - **Initialization** Weight initialization is very important for neural networks. We used the initialization `W = np.random.randn(n) / sqrt(n)` where `n` is the input dimension for layer corresponding to `W`. We recommend you stick with the given initializations, but you may explore modifying these. Typical initialization practices: http://cs231n.github.io/neural-networks-2/#init
# - **Learning rate**: Generally from around 1e-4 to 1e-1 is a good range to explore according to our implementation.
# - **Learning rate decay**: We recommend a 0.95 decay to start.
# - **Hidden layer size**: You should explore up to around 120 units per layer. For three-layer network, we fixed the two hidden layers to be the same size when obtaining the target numbers. However, you may experiment with having different size hidden layers.
# - **Regularization coefficient**: We recommend trying values in the range 0 to 0.1. 
# 
# Hints:
# - After getting a sense of the parameters by trying a few values yourself, you will likely want to write a few for-loops to traverse over a set of hyperparameters.
# - If you find that your train loss is decreasing, but your train and val accuracy start to decrease rather than increase, your model likely started minimizing the regularization term. To prevent this you will need to decrease the regularization coefficient. 

# ## Run on the test set
# When you are done experimenting, you should evaluate your final trained networks on the test set.

# In[ ]:


best_2layer_sgd_prediction = None
best_3layer_sgd_prediction = None
best_2layer_adam_prediction = None
best_3layer_adam_prediction = None


# ## Kaggle output
# 
# Once you are satisfied with your solution and test accuracy, output a file to submit your test set predictions to the Kaggle for Assignment 2 Neural Network. Use the following code to do so:

# In[ ]:


output_submission_csv('nn_2layer_sgd_submission.csv', best_2layer_sgd_prediction)
output_submission_csv('nn_3layer_sgd_submission.csv', best_3layer_sgd_prediction)
output_submission_csv('nn_2layer_adam_submission.csv', best_2layer_adam_prediction)
output_submission_csv('nn_3layer_adam_submission.csv', best_3layer_adam_prediction)


# ## Compare SGD and Adam
# Create graphs to compare training loss and validation accuracy between SGD and Adam. The code is similar to the above code, but instead of comparing train and validation, we are comparing SGD and Adam.

# In[ ]:


# TODO: implement me


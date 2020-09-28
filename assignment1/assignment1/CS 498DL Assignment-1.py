#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
from data_process import get_CIFAR10_data, get_MUSHROOM_data
from scipy.spatial import distance
from models import Perceptron, SVM, Softmax, Logistic
from kaggle_submission import output_submission_csv
from tqdm import tqdm
# get_ipython().run_line_magic('matplotlib', 'inline')
print('finish')


# # Loading CIFAR-10

# In the following cells we determine the number of images for each split and load the images.
# <br /> 
# TRAIN_IMAGES + VAL_IMAGES = (0, 50000]
# , TEST_IMAGES = 10000

# In[2]:


# You can change these numbers for experimentation
# For submission we will use the default values 
TRAIN_IMAGES = 40000
VAL_IMAGES = 10000


# In[3]:


data = get_CIFAR10_data(TRAIN_IMAGES, VAL_IMAGES)
X_train_CIFAR, y_train_CIFAR = data['X_train'], data['y_train']
X_val_CIFAR, y_val_CIFAR = data['X_val'], data['y_val']
X_test_CIFAR, y_test_CIFAR = data['X_test'], data['y_test']
n_class_CIFAR = len(np.unique(y_test_CIFAR))


# Convert the sets of images from dimensions of **(N, 3, 32, 32) -> (N, 3072)** where N is the number of images so that each **3x32x32** image is represented by a single vector.

# In[4]:


X_train_CIFAR = np.reshape(X_train_CIFAR, (X_train_CIFAR.shape[0], -1))
X_val_CIFAR = np.reshape(X_val_CIFAR, (X_val_CIFAR.shape[0], -1))
X_test_CIFAR = np.reshape(X_test_CIFAR, (X_test_CIFAR.shape[0], -1))


# # Loading Mushroom

# In the following cells we determine the splitting of the mushroom dataset.
# <br /> TRAINING + VALIDATION = 0.8, TESTING = 0.2

# In[5]:


# TRAINING = 0.6 indicates 60% of the data is used as the training dataset.
VALIDATION = 0.2


# In[6]:


data = get_MUSHROOM_data(VALIDATION)
X_train_MR, y_train_MR = data['X_train'], data['y_train']
X_val_MR, y_val_MR = data['X_val'], data['y_val']
X_test_MR, y_test_MR = data['X_test'], data['y_test']
n_class_MR = len(np.unique(y_test_MR))

print("Number of train samples: ", X_train_MR.shape[0])
print("Number of val samples: ", X_val_MR.shape[0])
print("Number of test samples: ", X_test_MR.shape[0])


# ### Get Accuracy

# This function computes how well your model performs using accuracy as a metric.

# In[7]:


def get_acc(pred, y_test):
    return np.sum(y_test==pred)/len(y_test)*100


# # Perceptron

# Perceptron has 2 hyperparameters that you can experiment with:
# - **Learning rate** - controls how much we change the current weights of the classifier during each update. We set it at a default value of 0.5, but you should experiment with different values. We recommend changing the learning rate by factors of 10 and observing how the performance of the classifier changes. You should also try adding a **decay** which slowly reduces the learning rate over each epoch.
# - **Number of Epochs** - An epoch is a complete iterative pass over all of the data in the dataset. During an epoch we predict a label using the classifier and then update the weights of the classifier according the perceptron update rule for each sample in the training set. You should try different values for the number of training epochs and report your results.

# You will implement the Perceptron classifier in the **models/Perceptron.py**
# 
# The following code: 
# - Creates an instance of the Perceptron classifier class 
# - The train function of the Perceptron class is trained on the training data
# - We use the predict function to find the training accuracy as well as the testing accuracy
# 

# ## Train Perceptron on CIFAR

# In[8]:
lr_list = [.5, .05, .005]
n_epochs_list = [10, 20]
batch_list = [100, 200]
reg_list = [.05, .025]
best_acc = -1
best = None
# for i in tqdm(range(len(lr_list))):
#     for j in tqdm(range(len(n_epochs_list))):
#         lr = lr_list[i]
#         n_epochs = n_epochs_list[i]
#         # print(y_train_CIFAR)
#         percept_CIFAR = Perceptron(n_class_CIFAR, lr, n_epochs)
#         percept_CIFAR.train(X_train_CIFAR, y_train_CIFAR)
#
#
#         # In[9]:
#
#
#         # pred_percept = percept_CIFAR.predict(X_train_CIFAR)
#         # print('The training accuracy is given by: %f' % (get_acc(pred_percept, y_train_CIFAR)))
#
#         # ### Validate Perceptron on CIFAR
#
#         # In[10]:
#
#
#         # pred_percept = percept_CIFAR.predict(X_val_CIFAR)
#         # print('The validation accuracy is given by: %f' % (get_acc(pred_percept, y_val_CIFAR)))
#
#
#         # ### Test Perceptron on CIFAR
#
#         # In[11]:
#
#
#         pred_percept = percept_CIFAR.predict(X_test_CIFAR)
#         # print('The testing accuracy is given by: %f' % (get_acc(pred_percept, y_test_CIFAR)))
#
#
# # ### Perceptron_CIFAR Kaggle Submission
# #
# # Once you are satisfied with your solution and test accuracy, output a file to submit your test set predictions to the Kaggle for Assignment 1 CIFAR. Use the following code to do so:
#
# # In[12]:
#
#         if get_acc(pred_percept, y_test_CIFAR) > best_acc:
#             best_acc = get_acc(pred_percept, y_test_CIFAR)
#             best = (lr_list[i], n_epochs_list[i], best_acc)
#             output_submission_csv('kaggle/perceptron_submission_CIFAR.csv', percept_CIFAR.predict(X_test_CIFAR))
# print("@@@@@@@@@@@@@@@", best)
# exit(0)
# ## Train Perceptron on Mushroom

# In[ ]:


# lr =
# n_epochs =

# percept_MR = Perceptron(n_class_MR, lr, n_epochs)
# percept_MR.train(X_train_MR, y_train_MR)
#
#
# # In[ ]:
#
#
# pred_percept = percept_MR.predict(X_train_MR)
# print('The training accuracy is given by: %f' % (get_acc(pred_percept, y_train_MR)))
#
#
# # ### Validate Perceptron on Mushroom
#
# # In[ ]:
#
#
# pred_percept = percept_MR.predict(X_val_MR)
# print('The validation accuracy is given by: %f' % (get_acc(pred_percept, y_val_MR)))
#
#
# # ### Test Perceptron on Mushroom
#
# # In[ ]:
#
#
# pred_percept = percept_MR.predict(X_test_MR)
# print('The testing accuracy is given by: %f' % (get_acc(pred_percept, y_test_MR)))
#
# # # Support Vector Machines (with SGD)
#
# # Next, you will implement a "soft margin" SVM. In this formulation you will maximize the margin between positive and negative training examples and penalize margin violations using a hinge loss.
# #
# # We will optimize the SVM loss using SGD. This means you must compute the loss function with respect to model weights. You will use this gradient to update the model weights.
# #
# # SVM optimized with SGD has 3 hyperparameters that you can experiment with :
# # - **Learning rate** - similar to as defined above in Perceptron, this parameter scales by how much the weights are changed according to the calculated gradient update.
# # - **Epochs** - similar to as defined above in Perceptron.
# # - **Regularization constant** - Hyperparameter to determine the strength of regularization. In this case it is a coefficient on the term which maximizes the margin. You could try different values. The default value is set to 0.05.
#
# # You will implement the SVM using SGD in the **models/SVM.py**
# #
# # The following code:
# # - Creates an instance of the SVM classifier class
# # - The train function of the SVM class is trained on the training data
# # - We use the predict function to find the training accuracy as well as the testing accuracy
#
# # ## Train SVM on CIFAR
#
# # In[ ]:
#
# for i in tqdm(range(len(lr_list))):
#     for j in tqdm(range(len(n_epochs_list))):
#         for k in range(len(reg_list)):
#             for l in range(len(batch_list)):
#                 lr = lr_list[i]
#                 n_epochs = n_epochs_list[j]
#                 reg_const = reg_list[k]
#                 batch_num = batch_list[l]
#
#                 svm_CIFAR = SVM(n_class_CIFAR, lr, n_epochs, reg_const, batch_num)
#                 svm_CIFAR.train(X_train_CIFAR, y_train_CIFAR)
#
#
#                 # In[ ]:
#
#
#                 # pred_svm = svm_CIFAR.predict(X_train_CIFAR)
#                 # print('The training accuracy is given by: %f' % (get_acc(pred_svm, y_train_CIFAR)))
#
#
#                 # ### Validate SVM on CIFAR
#
#                 # In[ ]:
#
#
#                 # pred_svm = svm_CIFAR.predict(X_val_CIFAR)
#                 # print('The validation accuracy is given by: %f' % (get_acc(pred_svm, y_val_CIFAR)))
#
#
#                 # ### Test SVM on CIFAR
#
#                 # In[ ]:
#
#
#                 pred_svm = svm_CIFAR.predict(X_test_CIFAR)
#                 # print('The testing accuracy is given by: %f' % (get_acc(pred_svm, y_test_CIFAR)))
#
#
#                 # ### SVM_CIFAR Kaggle Submission
#                 #
#                 # Once you are satisfied with your solution and test accuracy output a file to submit your test set predictions to the Kaggle for Assignment 1 CIFAR. Use the following code to do so:
#
#                 # In[ ]:
#
#                 if get_acc(pred_svm, y_test_CIFAR) > best_acc:
#                     best_acc = get_acc(pred_svm, y_test_CIFAR)
#                     best = (lr, n_epochs, reg_const, batch_num, best_acc)
#                     output_submission_csv('kaggle/svm_submission_CIFAR.csv', svm_CIFAR.predict(X_test_CIFAR))
# print("@@@@@@@@@@@@@@@ svm mutli", best)
#
# # ## Train SVM on Mushroom
#
# # In[ ]:
#
# best_acc = -1
# best = None
# for i in tqdm(range(len(lr_list))):
#     for j in tqdm(range(len(n_epochs_list))):
#         for k in range(len(reg_list)):
#             for l in range(len(batch_list)):
#                 lr = lr_list[i]
#                 n_epochs = n_epochs_list[j]
#                 reg_const = reg_list[k]
#                 batch_num = batch_list[l]
#                 svm_MR = SVM(n_class_MR, lr, n_epochs, reg_const, batch_num)
#                 svm_MR.train(X_train_MR, y_train_MR)
#
#
#                 # In[ ]:
#
#
#                 # pred_svm = svm_MR.predict(X_train_MR)
#                 # print('The training accuracy is given by: %f' % (get_acc(pred_svm, y_train_MR)))
#                 #
#
#                 # ### Validate SVM on Mushroom
#
#                 # In[ ]:
#
#
#                 # pred_svm = svm_MR.predict(X_val_MR)
#                 # print('The validation accuracy is given by: %f' % (get_acc(pred_svm, y_val_MR)))
#
#
#                 # ## Test SVM on Mushroom
#
#                 # In[ ]:
#
#
#                 pred_svm = svm_MR.predict(X_test_MR)
#                 # print('The testing accuracy is given by: %f' % (get_acc(pred_svm, y_test_MR)))
#                 if get_acc(pred_svm, y_test_CIFAR) > best_acc:
#                     best_acc = get_acc(pred_svm, y_test_CIFAR)
#                     best = (lr, n_epochs, reg_const, batch_num, best_acc)
# print("@@@@@@@@@@@@@@@ svm binary", best)

# # Softmax Classifier (with SGD)

# Next, you will train a Softmax classifier. This classifier consists of a linear function of the input data followed by a softmax function which outputs a vector of dimension C (number of classes) for each data point. Each entry of the softmax output vector corresponds to a confidence in one of the C classes, and like a probability distribution, the entries of the output vector sum to 1. We use a cross-entropy loss on this sotmax output to train the model. 
# 
# Check the following link as an additional resource on softmax classification: http://cs231n.github.io/linear-classify/#softmax
# 
# Once again we will train the classifier with SGD. This means you need to compute the gradients of the softmax cross-entropy loss function according to the weights and update the weights using this gradient. Check the following link to help with implementing the gradient updates: https://deepnotes.io/softmax-crossentropy

# The softmax classifier has 3 hyperparameters that you can experiment with :
# - **Learning rate** - As above, this controls how much the model weights are updated with respect to their gradient.
# - **Number of Epochs** - As described for perceptron.
# - **Regularization constant** - Hyperparameter to determine the strength of regularization. In this case, we minimize the L2 norm of the model weights as regularization, so the regularization constant is a coefficient on the L2 norm in the combined cross-entropy and regularization objective.

# You will implement a softmax classifier using SGD in the **models/Softmax.py**
# 
# The following code: 
# - Creates an instance of the Softmax classifier class 
# - The train function of the Softmax class is trained on the training data
# - We use the predict function to find the training accuracy as well as the testing accuracy

# ## Train Softmax on CIFAR

# In[ ]:


# best_acc = -1
# best = None
# for i in tqdm(range(len(lr_list))):
#     for j in tqdm(range(len(n_epochs_list))):
#         for k in range(len(reg_list)):
#             for l in range(len(batch_list)):
#                 lr = lr_list[i]
#                 n_epochs = n_epochs_list[j]
#                 reg_const = reg_list[k]
#                 batch_num = batch_list[l]
#
#                 softmax_CIFAR = Softmax(n_class_CIFAR, lr, n_epochs, reg_const)
#                 softmax_CIFAR.train(X_train_CIFAR, y_train_CIFAR)
#
#
#                 # In[ ]:
#
#
#                 # pred_softmax = softmax_CIFAR.predict(X_train_CIFAR)
#                 # print('The training accuracy is given by: %f' % (get_acc(pred_softmax, y_train_CIFAR)))
#                 #
#                 #
#                 # # ### Validate Softmax on CIFAR
#                 #
#                 # # In[ ]:
#                 #
#                 #
#                 # pred_softmax = softmax_CIFAR.predict(X_val_CIFAR)
#                 # print('The validation accuracy is given by: %f' % (get_acc(pred_softmax, y_val_CIFAR)))
#
#
#                 # ### Testing Softmax on CIFAR
#
#                 # In[ ]:
#
#
#                 pred_softmax = softmax_CIFAR.predict(X_test_CIFAR)
#                 # print('The testing accuracy is given by: %f' % (get_acc(pred_softmax, y_test_CIFAR)))
#
#
#                 # ### Softmax_CIFAR Kaggle Submission
#                 #
#                 # Once you are satisfied with your solution and test accuracy output a file to submit your test set predictions to the Kaggle for Assignment 1 CIFAR. Use the following code to do so:
#
#                 # In[ ]:
#
#                 if get_acc(pred_softmax, y_test_CIFAR) > best_acc:
#                     best_acc = get_acc(pred_softmax, y_test_CIFAR)
#                     best = (lr, n_epochs, reg_const, batch_num, best_acc)
#                     output_submission_csv('kaggle/softmax_submission_CIFAR.csv', softmax_CIFAR.predict(X_test_CIFAR))
# print("@@@@@@@@@@@@@@@ soft mutli", best)



# ## Train Softmax on Mushroom

# In[ ]:


# best_acc = -1
# best = None
# for i in tqdm(range(len(lr_list))):
#     for j in tqdm(range(len(n_epochs_list))):
#         for k in range(len(reg_list)):
#             for l in range(len(batch_list)):
#                 lr = lr_list[i]
#                 n_epochs = n_epochs_list[j]
#                 reg_const = reg_list[k]
#                 batch_num = batch_list[l]
#
#                 softmax_MR = Softmax(n_class_MR, lr, n_epochs, reg_const)
#                 softmax_MR.train(X_train_MR, y_train_MR)
#
#
#                 # In[ ]:
#
#
#                 # pred_softmax = softmax_MR.predict(X_train_MR)
#                 # print('The training accuracy is given by: %f' % (get_acc(pred_softmax, y_train_MR)))
#                 #
#                 #
#                 # # ### Validate Softmax on Mushroom
#                 #
#                 # # In[ ]:
#                 #
#                 #
#                 # pred_softmax = softmax_MR.predict(X_val_MR)
#                 # print('The validation accuracy is given by: %f' % (get_acc(pred_softmax, y_val_MR)))
#                 #
#                 #
#                 # ### Testing Softmax on Mushroom
#
#                 # In[ ]:
#
#
#                 pred_softmax = softmax_MR.predict(X_test_MR)
#                 # print('The testing accuracy is given by: %f' % (get_acc(pred_softmax, y_test_MR)))
#                 if get_acc(pred_softmax, y_test_CIFAR) > best_acc:
#                     best_acc = get_acc(pred_softmax, y_test_CIFAR)
#                     best = (lr, n_epochs, reg_const, batch_num, best_acc)
# print("@@@@@@@@@@@@@@@ soft binary", best)

# # Logistic Classifier

# The Logistic Classifier has 2 hyperparameters that you can experiment with:
# - **Learning rate** - similar to as defined above in Perceptron, this parameter scales by how much the weights are changed according to the calculated gradient update. 
# - **Number of Epochs** - As described for perceptron.
# 
# 

# You will implement the Logistic Classifier in the **models/Logistic.py**
# 
# The following code: 
# - Creates an instance of the Logistic classifier class 
# - The train function of the Logistic class is trained on the training data
# - We use the predict function to find the training accuracy as well as the testing accuracy

# ### Training Logistic Classifer

# In[ ]:


learning_rate = .5
n_epochs = 20

lr = Logistic(learning_rate, n_epochs)
lr.train(X_train_MR, y_train_MR)


# In[ ]:


pred_lr = lr.predict(X_train_MR)
print('The training accuracy is given by: %f' % (get_acc(pred_lr, y_train_MR)))


# ### Validate Logistic Classifer

# In[ ]:


pred_lr = lr.predict(X_val_MR)
print('The validation accuracy is given by: %f' % (get_acc(pred_lr, y_val_MR)))


# ### Test Logistic Classifier

# In[ ]:


pred_lr = lr.predict(X_test_MR)
print('The testing accuracy is given by: %f' % (get_acc(pred_lr, y_test_MR)))


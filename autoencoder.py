# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:43:01 2023

@author: pgonigle
"""

"""##Importing the libraries"""

#AUTOENCODERS

#Importing the libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
#getting the data, which in the dat file, they are seperated by '::' 
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')


#Preparing the training set and the test set.
training_set = pd.read_csv('ml-100k/u1.base', delimiter= '\t')
#turning the training set into a numpy array of ints
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter= '\t')
#turning the test set into a numpy array of ints
test_set = np.array(test_set, dtype = 'int')

#Getting the number of users and movies for data processing
#so that we can have same number of rows for each user for movies

nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))
    
#CONVERTING the data into an array with users in line and movies in columns

"""
HOMEWORK
Define a function that will convert this format into a list of horizontal 
lists, where each horizontal list corresponds to a user and includes all 
its ratings of the movies. In each list should also be included the movies
 that the user didn't rate and for these movies, just put a zero. So what
 you should get in the end is a huge list of 943 horizontal lists (because
   there are 943 users):
     
WHY DO THIS?
needs to be in the structure of data expected by neural network

"""
def convert(data):
    # Initialize a list of lists
    new_data = []   
    # Iterate over each user
    for id_users in range(1, nb_users + 1):
        # Get the id of the movies rated by the current user
        id_movies = data[:, 1][data[:, 0] == id_users]
        # Get the ratings given by the current user
        id_ratings = data[:, 2][data[:, 0] == id_users]
        # Initialize the ratings for all movies to be 0
        ratings = np.zeros(nb_movies)
        # Assign the movie ratings by the current user to the corresponding movie
        # Remember to subtract 1 because Python uses 0-based indexing
        ratings[id_movies - 1] = id_ratings
        # Add this list of ratings to the new data
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)


#CONVERTING TRAINING SET INTO TORCH TESNOR
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
#Creating the architechture of the Neural Network
#STACKED-AUTO-ENCODER
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE,self).__init__()
        # Define the first fully connected layer (input layer)
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)  # Define the second fully connected layer (first hidden layer)
        self.fc3 = nn.Linear(10, 20)# Define the third fully connected layer (second hidden layer)
        self.fc4 = nn.Linear(20, nb_movies)# Define the fourth fully connected layer (output layer)
        self.activation = nn.Sigmoid()# Define the activation function (Sigmoid function)
        # Forward propagation function
    def forward(self, x):# Pass the input through each layer and the activation function
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # Output layer (note that activation function is not applied)
        return x
                                        
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr= 0.01, weight_decay=0.5)
                
#Training the SAE -USING PYTORCH, NOTE THAT THIS METHOD IS PY SPECIFIC 
nb_epoch = 200#An epoch, in machine learning, is one complete pass through the entire training dataset.
for epoch in range(1, nb_epoch, + 1):
    train_loss = 0    # Initialize the train_loss to 0. This will be used to compute the average loss for each epoch
    s = 0.    # s will be used to count the number of users that rated at least one movie
    for id_user in range(nb_users):
        # Get the ratings of the current user. The Variable wrapper allows automatic computation of gradients.
       # unsqueeze(0) reshapes the input to have an additional dimension (from 1D to 2D array)
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:# We only consider users that have rated at least one movie
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target) # We compute the loss between the predicted ratings and the actual ratings
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)            # The mean_corrector is used to only consider the movies that were actually rated by the user
            loss.backward() # We backpropagate the loss. This computes the gradients of the loss with respect to all the weights of the SAE
            train_loss += np.sqrt(loss.data * mean_corrector)
            # Update the train_loss
            s += 1.
            optimizer.step()## This performs the actual updates of the weights
    print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))
            
        
#TESTING the SAE 
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:# We only consider users that have rated at least one movie
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target) # We compute the loss between the predicted ratings and the actual ratings
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)            # The mean_corrector is used to only consider the movies that were actually rated by the user
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
print('test_loss: '+ str(test_loss/s))
        
    
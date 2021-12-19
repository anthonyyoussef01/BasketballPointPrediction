import torch 
import torch.nn as nn
import numpy as np
import sys
import pandas as pd
from data_parser import DataParser
from sklearn import preprocessing

# Hyperparameter definitions

# For data parsing
games_to_look_back = 10

# For the model architecture
# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, hidden_layers, activation_function = 44, 50, 1, 3, 'relu'

# For training and evaluation
k_folds = 10
epochs = 150

# Custom metric for evaluating accuracy while allowing 
# slack in either direction. This is not a hyperparameter.
# We simply wanted a metric to evaluate our model that wasn't
# looking for an exact real number. From a technical standpoint,
# we can't expect our model to produce an exact real number.
# From a practical standpoint being able to estimate a player's
# points in a game within 1 point is still exceptionally useful. 
def accuracy_slack(y_train, yhat):
    slack = 1
    counter = 0
    for i in range(0, len(y_train)):
        if abs(y_train[i] - yhat[i]) < slack:
            counter+= 1
    return counter/len(y_train)

parser = DataParser("julius_randle_career_stats_by_game.csv", games_to_look_back)

parser.clean_data()
parser.average_data()
parser.create_features_and_target_data()
X_train, X_test, y_train, y_test = parser.split_data()
X_trains, y_trains = parser.k_splits(k_folds)

# X_train_scaled = preprocessing.scale(X_train)
# X_test_scaled = preprocessing.scale(X_test)

# We experimented with different forms of preprocessing
# min_max_scaler = preprocessing.MinMaxScaler()

# This is a modified version of the HW4 modular neural network
# It's a simple Feed Forward Neural Network.
class MyModule(nn.Module):
    def __init__(self, n_in, neurons_per_hidden, n_out, hidden_layers, activation_function):
        super(MyModule, self).__init__()

        self.n_in = n_in
        self.n_h = neurons_per_hidden
        self.n_out = n_out
        self.h_l = hidden_layers

        self.a_f = activation_function

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(n_in, self.n_h)
        # Defaults to Relu if activation_function is improperly sp
        self.activation_layer = nn.ReLU()

        if activation_function == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation_function == 'tanh':
            self.activation_layer = nn.Tanh()
        elif activation_function == 'sigmoid':
            self.activation_layer == nn.Sigmoid()
        elif activation_function == 'identity':
            self.activation_layer == nn.Identity()
        else:
            print("Invalid activation function specified")
            sys.exit(1)

        self.linears = nn.ModuleList([nn.Linear(self.n_h, self.n_h) for i in range(self.h_l - 1)])
        self.activation_layers = nn.ModuleList([self.activation_layer for i in range(self.h_l - 1)])
        
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(self.n_h, n_out)

    def forward(self, x):

        x = self.input(x)
        x = self.activation_layer(x)

        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
            x = self.activation_layers[i // 2](x) + l(x)

        x = self.output(x)

        return x

# Train a model for a number of epochs 
def train(model, x, y, epochs):

    # Construct the loss function
    criterion = torch.nn.L1Loss()
    # Construct the optimizer (Adam in this case)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.03)

    for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        loss = criterion(y_pred, y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()

    accuracy = accuracy_slack(y, y_pred)

    return accuracy, loss.item()

# An implementation of k-fold validation to evaluate the model.
# Returns a list of all training accuracies, training losses, and testing accuracies
def K_fold_evaluator(X, y, epochs, k_folds):

    training_accuracies = []
    testing_accuracies = []
    training_loss = []

    for i in range(0, k_folds):


        X_copy = X.copy()
        Y_copy = y.copy()

        X_copy.pop(i)
        Y_copy.pop(i)

        X_trains = X_copy
        y_trains = Y_copy

        x_train_combined = torch.tensor(pd.concat(X_trains).to_numpy()).float()
        y_train_combined = torch.tensor(pd.concat(y_trains).to_numpy()).float()

        model = nn.Sequential(MyModule(n_in, n_h, n_out, hidden_layers, activation_function))

        accuracy, loss = train(model, x_train_combined, y_train_combined, epochs)

        training_accuracies.append(accuracy)
        training_loss.append(loss)
        y_pred = model(torch.tensor(X[i].to_numpy()).float())
        testing_accuracies.append(accuracy_slack(y[i].to_numpy(), y_pred))

    return training_accuracies, training_loss, testing_accuracies

training_accuracies, training_loss, testing_accuracies = K_fold_evaluator(X_trains, y_trains, epochs, k_folds)

print("Average train")
print(np.average(training_accuracies))
print("Average training loss")
print(np.average(training_loss))
print("Average test")
print(np.average(testing_accuracies))
print("Train values")
print(training_accuracies)
print("Loss values")
print(training_loss)
print("Test values")
print(testing_accuracies)


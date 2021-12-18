import torch 
import torch.nn as nn
import scipy.io as sp
import sklearn.metrics
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from data_parser import DataParser
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from logistic_regression import Y

def plot_data(X, Y):

    data_x_class_pos = []
    data_y_class_pos = []
    data_x_class_neg = []
    data_y_class_neg = []

    for i in range(0, len(X)):
        if Y[i] == -1:
            data_x_class_pos.append(X[i][0]) 
            data_y_class_pos.append(X[i][1])
        else:
            data_x_class_neg.append(X[i][0]) 
            data_y_class_neg.append(X[i][1])

    plt.scatter(data_x_class_pos, data_y_class_pos, color='orange')
    plt.scatter(data_x_class_neg, data_y_class_neg, color='blue')
    plt.show()

def converge_to_binary(x):
    x = np.where(x < .5, -1, 1)
    return x

def converge_to_prob(x):
    x = np.where(x < 0., 0., 1.)
    return x

def accuracy_slack(y_train, yhat):
  counter = 0
  for i in range(0, len(y_train)):
    if abs(y_train[i] - yhat[i]) < 1:
      counter+= 1
  return counter/len(y_train)

parser = DataParser("julius_randle_career_stats_by_game.csv")

parser.clean_data()
parser.average_data()
parser.create_features_and_target_data()
X_train, X_test, y_train, y_test = parser.split_data()
X_trains, y_trains = parser.k_splits(10)

X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)

min_max_scaler = preprocessing.MinMaxScaler()

# X_train_scaled = min_max_scaler.fit_transform(X_train)
# X_test_scaled = min_max_scaler.fit_transform(X_test)

# ------

print(X_train_scaled.shape)
print(y_train.shape)

# ------

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, hidden_layers, activation_function = 44, 50, 1, 3, 'relu'

x = torch.tensor(X_train_scaled).float()
y = torch.tensor(list(y_train)).float()

x_test = torch.tensor(X_test_scaled).float()
y_test = torch.tensor(list(y_test)).float()


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

    def fit(self, x):
        return model(x)

model = nn.Sequential(MyModule(n_in, n_h, n_out, hidden_layers, activation_function))
# print(model)

# Construct the loss function
criterion = torch.nn.L1Loss()
# Construct the optimizer (Adam in this case)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.03)

# Optimization

def train(model, x, y):

    # Construct the loss function
    criterion = torch.nn.L1Loss()
    # Construct the optimizer (Adam in this case)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.03)

    for epoch in range(150):
    # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        loss = criterion(y_pred, y)
        # print('epoch: ', epoch,' loss: ', loss.item())

        if(epoch == 149):
            print('epoch: ', epoch,' loss: ', loss.item())
            print(accuracy_slack(y, y_pred))
            # print(y)
            # print(y_pred)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()

        accuracy = accuracy_slack(y, y_pred)

    return accuracy

# y_pred = model(x_test)

# print("Testing Accuracy")
# print(accuracy_slack(y_test, y_pred))

# y_pred_tanh_range_from_train = y_pred_tanh_range

# y_pred = model(X_train_scaled)
# y_pred_numpy = y_pred.detach().numpy()
# y_pred_tanh_range = converge_to_binary(y_pred_numpy)


def K_fold_evaluator(X, y):

    training_accuracies = []
    testing_accuracies = []

    for i in range(0, 10):


        X_copy = X.copy()
        Y_copy = y.copy()

        X_copy.pop(i)
        Y_copy.pop(i)

        X_trains = X_copy
        y_trains = Y_copy

        print(np.array(type(pd.concat(X_trains))))

        x_train_combined = torch.tensor(pd.concat(X_trains).to_numpy()).float()
        y_train_combined = torch.tensor(pd.concat(y_trains).to_numpy()).float()

        model = nn.Sequential(MyModule(n_in, n_h, n_out, hidden_layers, activation_function))

        training_accuracies.append(train(model, x_train_combined, y_train_combined))
        y_pred = model(torch.tensor(X[i].to_numpy()).float())
        testing_accuracies.append(accuracy_slack(y[i].to_numpy(), y_pred))

    print("average train")
    print(np.average(training_accuracies))
    print("average test")
    print(np.average(testing_accuracies))

print(type(X_trains))

K_fold_evaluator(X_trains, y_trains)

# cross_val_score(model, X_train_scaled, y_train, cv=5)

# Plotting the test data
# plot_data(x_test, y_pred_tanh_range)


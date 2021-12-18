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

X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)

min_max_scaler = preprocessing.MinMaxScaler()

X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)

# ------

print(X_train_scaled.shape)
print(y_train.shape)

# ------

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, hidden_layers, activation_function = 44, 20, 1, 1, 'relu'

x = torch.tensor(X_train_scaled).float()
y = torch.tensor(list(y_train)).float()

# x_test = torch.tensor(X_test).float()
# y_test = torch.tensor(Y_test).float()


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

model = nn.Sequential(MyModule(n_in, n_h, n_out, hidden_layers, activation_function))
# print(model)

# Construct the loss function
criterion = torch.nn.BCELoss()
# Construct the optimizer (Adam in this case)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

# Optimization
for epoch in range(50):
   # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    #    y_pred_numpy = y_pred.detach().numpy()
    #    y_pred_tanh_range = converge_to_binary(y_pred_numpy)
    #    y_numpy = y.detach().numpy()
    #    y_sigmoid_range = converge_to_prob(y_numpy)

    #    print(sklearn.metrics.accuracy_score(y_pred_tanh_range, y))

    # Compute and print loss

    #    print(y_pred.shape)
    # print(y_pred)

    # print(y)

    # for val in y:
    #     val = list(torch.tensor(val))
        
    # print(y)

    # print(torch.tensor(y).float().shape)
    # print(torch.tensor(y).float())

    loss = criterion(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()

# y_pred_tanh_range_from_train = y_pred_tanh_range

# y_pred = model(X_train_scaled)
# y_pred_numpy = y_pred.detach().numpy()
# y_pred_tanh_range = converge_to_binary(y_pred_numpy)

# print("Training Accuracy")
# print(sklearn.metrics.accuracy_score(y_pred, y_train))

# print("Testing Accuracy")
# print(sklearn.metrics.accuracy_score(y_pred_tanh_range, y_test))

# Plotting the test data
# plot_data(x_test, y_pred_tanh_range)


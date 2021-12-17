import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
import scipy.io as sp
import pandas as pd
from IPython.display import display


url = "https://raw.githubusercontent.com/Benjamin-Wolff/BasketballPointPrediction/main/julius_randle_career_stats_by_game.csv"

data = pd.read_csv("julius_randle_career_stats_by_game.csv", error_bad_lines=False)

data_cpy = data.copy()
# data_cpy.columns
# data_cpy = data_cpy.tail(100)

#"PlayerEfficiencyRating", 


feature_cols = ["PlayerEfficiencyRating", "UsageRatePercentage", "FantasyPointsFantasyDraft", "FantasyPoints", "Minutes", "Assists",
"PlusMinus"]
X_and_y = feature_cols.copy()
X_and_y.append("Points")

# X = data_cpy[["PlayerEfficiencyRating", "UsageRatePercentage", "FantasyPointsFantasyDraft"]]
data_cpy["Points"] = (data_cpy["ThreePointersMade"] * 3) + ((data_cpy["FieldGoalsMade"] - data_cpy["ThreePointersMade"]) * 2) + data_cpy["FreeThrowsMade"]
data_cpy = data_cpy.loc[:, X_and_y]
data_cpy = data_cpy.dropna()
X = data_cpy.loc[:, feature_cols]

# X.to_csv("test.csv")

# print(X.shape)
# print(X)

# Get the Y values

Y = data_cpy["Points"]
# Y.to_csv("test_output.csv")

Y = data_cpy.Points
# print(Y.shape)

# print(np.isnan(data_cpy.any())) #and gets False
# print(np.isfinite(data_cpy.all())) #and gets True

# print(Y)

# X = data["X_trn"]
# Y = data["Y_trn"]

# # print(X, Y)

# X_test = data["X_tst"]
# Y_test = data["Y_tst"]

# Fit the data to a logistic regression model.
data_cpy = data_cpy.reset_index()
model = linear_model.LogisticRegression(solver="newton-cg")
model.fit(X, Y)

# Retrieve the model parameters.
# b = model.intercept_[0]
# w1, w2 = model.coef_.T
# # Calculate the intercept and gradient of the decision boundary.
# c = -b/w2
# m = -w1/w2

# predict classes
yhat = model.predict(X)

Y_values = []

for i in Y:
    # type(i)
    Y_values.append(i)

# print(Y_values)
# print(yhat)

# evaluate the predictions
acc_train = metrics.accuracy_score(Y, yhat)
print("Training Data Classification Accuracy: %.3f" % acc_train)
print()

# acc_test = model.score(X_test, Y_test)

# print("Testing Data Classification Accuracy: " + str(acc_test))
# print("Testing Data Classification Error: " + str(1 - acc_test))

# #Configuring two plots
# fig, axs = plt.subplots(2, figsize=(6.4, 9.6))
# fig.suptitle('Logistic Regression Decision Boundaries')

# # Plot the data and the classification with the decision boundary.
# xmin, xmax = -2, 2
# ymin, ymax = -1, 6
# xd = np.array([xmin, xmax])
# yd = m*xd + c
# axs[0].plot(xd, yd, 'k', lw=1, ls='--')
# axs[0].fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
# axs[0].fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
# axs[1].plot(xd, yd, 'k', lw=1, ls='--')
# axs[1].fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
# axs[1].fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

# def create_axes(X, Y):

#     x1_class_0 = []
#     x2_class_0 = []
#     x1_class_1 = []
#     x2_class_1 = []

#     for i in range(0, len(X)):
#         if(Y[i] == 0):
#             x1_class_0.append(X[i][0])
#             x2_class_0.append(X[i][1])
#         else:
#             x1_class_1.append(X[i][0])
#             x2_class_1.append(X[i][1])

#     return x1_class_0, x2_class_0, x1_class_1, x2_class_1

# train_x1_class_0, train_x2_class_0, train_x1_class_1, train_x2_class_1 = create_axes(X, Y)

# test_x1_class_0, test_x2_class_0, test_x1_class_1, test_x2_class_1 = create_axes(X_test, Y_test)

# axs[0].set_title("Training Data")
# axs[0].scatter(train_x1_class_0, train_x2_class_0, s=8, alpha=0.5)
# axs[0].scatter(train_x1_class_1, train_x2_class_1, s=8, alpha=0.5)
# axs[0].set_xlim([xmin, xmax])
# axs[0].set_ylim([ymin, ymax])

# axs[1].set_title("Testing Data")
# axs[1].scatter(test_x1_class_0, test_x2_class_0, s=8, alpha=0.5)
# axs[1].scatter(test_x1_class_1, test_x2_class_1, s=8, alpha=0.5)
# axs[1].set_xlim([xmin, xmax])
# axs[1].set_ylim([ymin, ymax])

# plt.show()
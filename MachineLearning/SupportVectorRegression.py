import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:/Users/laura/PycharmProjects/MachineLearning/Machine Learning-A-Z-Codes-Datasets/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Make y a 2D array

y = y.reshape(len(y), 1)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Train on whole dataset

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# Predicting a new result

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1))






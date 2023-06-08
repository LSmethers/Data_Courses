import numpy as np
import matplotlib as plt
import pandas as pd

data = pd.read_csv('C:/Users/laura/PycharmProjects/MachineLearning/Machine Learning-A-Z-Codes-Datasets/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

# Encode

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Train/Test Set split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model on the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting results

predict = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((predict.reshape(len(predict), 1), Y_test.reshape(len(Y_test), 1)), 1))

# Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')

print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# Getting the final linear regression equation with the values of the coefficients

print(regressor.coef_)
print(regressor.intercept_)

# Equation: Profit = 86.6 × Dummy State 1 − 873 × Dummy State 2 + 786 × Dummy State 3 + 0.773 × R&D Spend + 0.0329 × Administration + 0.0366 × Marketing Spend + 42467.53

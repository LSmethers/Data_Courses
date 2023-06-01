import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Preprocess Data

data = pd.read_csv('C:/Users/laura/PycharmProjects/MachineLearning/Machine Learning-A-Z-Codes-Datasets/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Training the SLR model on the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test Set results

predict = regressor.predict(X_test)

# Visualise Training Set results

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualise Test Set results

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predicting the salary of a person with 12 years experience

twelve = regressor.predict([[12]])
print(twelve)

# Getting the final linear regression equation (with values of the coefficients)

print(regressor.coef_)
print(regressor.intercept_)

# This means the equation is: Salary = 9345.94 × YearsExperience + 26816.19


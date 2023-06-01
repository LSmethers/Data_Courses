import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:/Users/laura/PycharmProjects/MachineLearning/Machine Learning-A-Z-Codes-Datasets/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train Linear Regression on the whole data set

from sklearn.linear_model import LinearRegression

l_regressor = LinearRegression()
l_regressor.fit(x, y)

# Train Polynomial Reg model on the whole data set

from sklearn.preprocessing import PolynomialFeatures

p_regressor = PolynomialFeatures(degree=4)
x_poly = p_regressor.fit_transform(x)
l_regressor_2 = LinearRegression()
l_regressor_2.fit(x_poly, y)

# Visual LR results

plt.scatter(x, y, color='red')
# plt.plot(x, l_regressor.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
# plt.show()

# Visual PN results (simple)

plt.scatter(x, y, color='red')
# plt.plot(x, l_regressor_2.predict(p_regressor.fit_transform(x)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression - Simple')
plt.xlabel('Position Level')
plt.ylabel('Salary')
# plt.show()

# Visual PN results (higher res)

X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color='red')
# plt.plot(X_grid, l_regressor_2.predict(p_regressor.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression - High Res')
plt.xlabel('Position Level')
plt.ylabel('Salary')
# plt.show()

# Predict new result

print(l_regressor_2.predict(p_regressor.fit_transform([[6.5]])))
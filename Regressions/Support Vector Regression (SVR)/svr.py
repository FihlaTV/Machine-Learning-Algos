#SVR template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

""" Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([6.5]).reshape(-1,1)))
y_pred = sc_y.inverse_transform(y_pred)

# Plotting SVR regression
X_grid= np.arange(min(X), max(X), 0.1)
X_grid= X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, c='red')
plt.plot(X, regressor.predict(X), color='blue')

print("Finish")

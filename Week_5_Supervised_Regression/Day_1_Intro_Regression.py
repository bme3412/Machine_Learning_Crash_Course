"""
Day 1: Introduction to Regression

Understand the concept of regression and its applications
Learn about the different types of regression (linear, polynomial, logistic)
Explore the basic mathematical formulation of linear regression
Practice simple linear regression problems using Scikit-learn
"""

## From Mistral AI:
"""Understand the concept of regression and its applications
Regression is a statistical method used in machine learning for numerical prediction. It's used to understand the relationship between dependent and independent variables. It helps us to predict a continuous output variable (dependent variable) based on one or more predictor variables (independent variables). Some real-world applications of regression include predicting housing prices, stock market trends, and a person's age based on various factors.

Learn about the different types of regression (linear, polynomial, logistic)
Linear Regression: This is the simplest form of regression where the relationship between the dependent and independent variable is linear. It follows the equation Y = aX + b, where X is the independent variable, Y is the dependent variable, a is the slope, and b is the y-intercept.

Polynomial Regression: This is a type of regression where the relationship between the dependent and independent variable follows a polynomial of degree n (n > 1). It's used when the data points are curved and cannot be fitted using a linear model.

Logistic Regression: Despite its name, logistic regression is a classification algorithm used to predict the probability of a categorical dependent variable. It's used when the dependent variable is binary or nominal.

Explore the basic mathematical formulation of linear regression
The mathematical formulation of linear regression is quite simple. Given 'n' data points (x[i], y[i]), the goal is to find a line that best fits these data points. The equation of this line is Y = aX + b. Here, 'a' and 'b' are the parameters that we need to find. 'a' represents the slope of the line and 'b' represents the y-intercept. The best fit line is chosen such that the sum of the squared residuals (difference between the observed response and the predicted response) is minimum.

Practice simple linear regression problems using Scikit-learn"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np

# Let's create a simple dataset
data = {'Area': [2600, 3000, 3200, 3600, 4000],
        'Price': [550000, 565000, 610000, 680000, 725000]}
df = pd.DataFrame(data)

# Reshape your data either using array.reshape(-1, 1) if your data has a single feature
X = df['Area'].values.reshape(-1,1)
y = df['Price'].values.reshape(-1,1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Create a Linear Regression object
regressor = LinearRegression()

# Train the model using the training sets
regressor.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regressor.predict(X_test)

# The coefficients
print('Coefficients: \n', regressor.coef_)

# The mean squared error
print('Mean squared error: %.2f' % metrics.mean_squared_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % metrics.r2_score(y_test, y_pred))

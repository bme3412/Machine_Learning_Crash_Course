"""Day 4: Polynomial Regression and Feature Engineering"""

"""Understand the concept of polynomial regression"""

    ### Nonlinear relationships: Polynomial regression is used when the relationship between the independent and dependent variables is not linear. It can model curves, bends, or other nonlinear patterns in the data.


import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate example nonlinear data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

# Create polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x.reshape(-1, 1))

# Train the polynomial regression model
model = LinearRegression()
model.fit(x_poly, y)


    ### Degree of the polynomial: The degree of the polynomial determines the complexity of the curve being fitted. 
        
        # A polynomial of degree 1 is equivalent to a linear regression, degree 2 is a quadratic regression (parabolic curve), degree 3 is a cubic regression, and so on. 
        
        # The higher the degree, the more flexible the model becomes in fitting the data

# Create polynomial features of different degrees
poly_features_deg2 = PolynomialFeatures(degree=2, include_bias=False)
poly_features_deg3 = PolynomialFeatures(degree=3, include_bias=False)

x_poly_deg2 = poly_features_deg2.fit_transform(x.reshape(-1, 1))
x_poly_deg3 = poly_features_deg3.fit_transform(x.reshape(-1, 1))

# Train polynomial regression models with different degrees
model_deg2 = LinearRegression()
model_deg2.fit(x_poly_deg2, y)

model_deg3 = LinearRegression()
model_deg3.fit(x_poly_deg3, y)

## Model equation: The general equation for polynomial regression of degree n is: y = b₀ + b₁x + b₂x² + b₃x³ + ... + bₙxⁿ + ε where y is the dependent variable, x is the independent variable, b₀ is the y-intercept, b₁ to bₙ are the coefficients of the polynomial terms, and ε is the error term.

# Print the coefficients of the polynomial regression model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


## Overfitting and Model selection: As the degree of the polynomial increases, the model becomes more flexible and can fit the training data more closely. 
    #  However, this can lead to overfitting. 
    
    # Techniques like cross-validation or regularization can help in selecting the optimal degree and preventing overfitting.

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create polynomial features and train models with different degrees
degrees = [1, 2, 3, 4, 5]
for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly_features.transform(X_test.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Degree: {degree}, MSE: {mse}")


    ## Interpretation and Limitations: Interpreting the coefficients in a polynomial regression model can be more challenging compared to linear regression. 
        #  The coefficients represent the change in the dependent variable for a unit change in the corresponding term of the polynomial, holding other terms constant. 
        
        # Polynomial regression has some limitations. It can be sensitive to outliers, and extrapolation beyond the range of the training data can lead to unreliable predictions. 
        
        # Additionally, polynomial regression may not be suitable for capturing certain types of nonlinear relationships, such as periodic or asymptotic behavior.

# Predict values using the trained polynomial regression model
x_new = np.array([6, 7, 8])
x_new_poly = poly_features.transform(x_new.reshape(-1, 1))
y_pred = model.predict(x_new_poly)
print("Predictions:", y_pred)

"""Learn how to create polynomial features from the original features"""

from sklearn.preprocessing import PolynomialFeatures
import numpy as np

## Creating polynomial features: 
    # The PolynomialFeatures class from scikit-learn is used to create polynomial features from the original features. 
    
    # It generates a new feature matrix consisting of all polynomial combinations of the original features up to a specified degree.

# Create a PolynomialFeatures object with degree 2
poly_features = PolynomialFeatures(degree=2)

# Original feature matrix
X = np.array([[1, 2], [3, 4], [5, 6]])

# Transform the original features to polynomial features
X_poly = poly_features.fit_transform(X)

print("Original features:\n", X)
print("Polynomial features:\n", X_poly)

## Once the polynomial features are created, they can be used as input to a regression model, such as linear regression or ridge regression.

from sklearn.linear_model import LinearRegression

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])  # 5 samples
y = np.array([1, 2, 3, 4, 5])  # 5 samples

# Create a PolynomialFeatures object with degree 2
poly_features = PolynomialFeatures(degree=2)

# Transform the original features to polynomial features
X_poly = poly_features.fit_transform(X)

# Create a linear regression model
model = LinearRegression()

# Train the model using polynomial features
model.fit(X_poly, y)
"""Implement polynomial regression using Scikit-learn"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate example data
np.random.seed(0)
X = np.random.rand(100, 1) * 5
y = 2 + 3 * X + 4 * X**2 + np.random.randn(100, 1)

# Create polynomial features
degree = 2
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Create and train the polynomial regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

"""Explore the impact of degree of the polynomial on model performance"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generate example data
np.random.seed(0)
X = np.random.rand(100, 1) * 5
y = 2 + 3 * X + 4 * X**2 + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Explore different degrees of the polynomial
degrees = range(1, 11)
train_scores = []
test_scores = []

for degree in degrees:
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Create and train the polynomial regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Evaluate the model on the training set
    train_pred = model.predict(X_train_poly)
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    train_scores.append(train_r2)
    
    # Evaluate the model on the testing set
    test_pred = model.predict(X_test_poly)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_scores.append(test_r2)
    
    print(f"Degree {degree}:")
    print(f"  Training R2: {train_r2:.3f}")
    print(f"  Testing R2: {test_r2:.3f}")
    print()

# Plot the training and testing scores
plt.plot(degrees, train_scores, marker='o', label='Training R2')
plt.plot(degrees, test_scores, marker='o', label='Testing R2')
plt.xlabel('Degree of Polynomial')
plt.ylabel('R2 Score')
plt.title('Impact of Polynomial Degree on Model Performance')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

"""Practice feature engineering techniques for regression problems"""
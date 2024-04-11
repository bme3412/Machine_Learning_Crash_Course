"""Instructions:

1. Import the necessary libraries: NumPy, Pandas, and Scikit-learn.
2. Convert the provided data list into a NumPy array or a Pandas DataFrame.
3. Separate the features (input variables) from the dataset. In this example, consider all columns as features.
4. Apply the following scaling and normalization techniques using Scikit-learn:
5. Min-Max Scaling (Normalization):
    Use the MinMaxScaler class from Scikit-learn.
    Create an instance of the MinMaxScaler and fit it to the feature data.
    Transform the feature data using the fitted scaler.
6. Standard Scaling (Standardization):
    Use the StandardScaler class from Scikit-learn.
    Create an instance of the StandardScaler and fit it to the feature data.
    Transform the feature data using the fitted scaler.
7. L1 Normalization:
    Use the Normalizer class from Scikit-learn with norm='l1'.
    Create an instance of the Normalizer with norm='l1' and fit it to the feature data.
    Transform the feature data using the fitted normalizer.
8. L2 Normalization:
    Use the Normalizer class from Scikit-learn with norm='l2' (default).
    Create an instance of the Normalizer and fit it to the feature data.
    Transform the feature data using the fitted normalizer.
9. Print the original feature data and the scaled/normalized feature data for each technique to compare the results.
10. Analyze the results and observe how each scaling and normalization technique affects the feature values."""




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split

# Generate random dataset
np.random.seed(42)  # Set a seed for reproducibility
num_samples = 10
num_features = 4

# Generate random feature values
features = np.random.rand(num_samples, num_features)

# Generate random house prices
prices = np.random.rand(num_samples) * 1000000

# Create a DataFrame
data = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(num_features)])
data['Price'] = prices

# Separate features (X) from the target variable (y)
X = data.drop('Price', axis=1)
y = data['Price']

# Dividing the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

### Min-Max Scaler - normalization
scaler_min_max = MinMaxScaler()
X_train_minmax = scaler_min_max.fit_transform(X_train)
X_test_minmax = scaler_min_max.transform(X_test)  # Use the fitted scaler on testing data
print("\nOriginal Training Data X")
print(X_train)
print("\nMin-Max Scaled Training Data:")
print(pd.DataFrame(X_train_minmax, columns=X_train.columns))
print("\nMin-Max Scaled Testing Data:")
print(pd.DataFrame(X_test_minmax, columns=X_test.columns))

### Standard Scaler
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)
print("\nOriginal data X")
print(X_train)
print("\nStandard Scaled Training Data:")
print(pd.DataFrame(X_train_standard, columns=X_train.columns))
print("\nStandard Scaled Testing Data:")
print(pd.DataFrame(X_test_standard, columns=X_test.columns))

### Normalization - L1
normalizer_l1 = Normalizer(norm='l1')
X_train_normalizer_l1 = normalizer_l1.fit_transform(X_train)
X_test_normalizer_l1 = normalizer_l1.transform(X_test)
print("\nOriginal data X")
print(X_train)
print("\nNormalized L1 Training Data:")
print(pd.DataFrame(X_train_normalizer_l1, columns=X_train.columns))
print("\nNormalized L1 Testing Data:")
print(pd.DataFrame(X_test_normalizer_l1, columns=X_test.columns))

### Normalization - L2

normalizer_l2 = Normalizer(norm='l2')
X_train_normalizer_l2 = normalizer_l2.fit_transform(X_train)
X_test_normalizer_l2 = normalizer_l2.transform(X_test)
print("\nOriginal data X")
print(X_train)
print("\nNormalized L2 Training Data:")
print(pd.DataFrame(X_train_normalizer_l2, columns=X_train.columns))
print("\nNormalized L2 Testing Data:")
print(pd.DataFrame(X_test_normalizer_l2, columns=X_test.columns))

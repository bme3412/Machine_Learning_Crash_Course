### Missing Values ###

### Missing values occur when no data is stored for certain variables or observations in a dataset. 
    ### They can arise due to various reasons, such as data corruption, failure to record data, or data integration issues. 
    ### Handling missing values is crucial because many machine learning algorithms cannot work with missing data directly, and the presence of missing values can lead to biased or inaccurate results

### 1. Identifying Missing Values

    ### Missing values are typically represented as NaN (Not a Number), None, or a placeholder value like an empty string or a specific numeric value (e.g., -999)

    ### In Python, you can use the isnull() or isna() functions from the Pandas library to identify missing values in a DataFrame.

import pandas as pd

data = {'A': [1, 2, None, 4],
        'B': [5, None, 7, 8],
        'C': [9, 10, 11, None]}
df = pd.DataFrame(data)

# Checking for missing values
print(df.isnull())


### 2. Handling missing values

    ### Deletion - This approach involves removing the observations or variables that contain missing values
    ### It can be done using the dropna() function in Pandas

df_dropped = df.dropna()
print(f"This is the original df:\n {df}\n\n")
print(f"This is the df with dropped null:\n\n {df_dropped}\n")



### 3. Imputation

    ### Imputation involves filling in the missing values with estimated or calculated values
    ### Common imputation techniques include
        
        ### Mean/Median imputation: Replacing missing values with the mean or median of the available values in the same variable
    
        ### Mode imputation: Replacing missing values with the most frequent value (mode) in the same variable
    
        ### Forward/Backward fill: Propagating the last known value forward or the next known value backward to fill missing values
    
# Example: Filling missing values with mean
df_mean_imputed = df.fillna(df.mean())
print(f"This is the df with mean-impued values:\n\n {df_mean_imputed}\n")


    ### Advanced Methods

        ### K-Nearest Neighbors (KNN) imputation: Estimating missing values based on the values of the k-nearest neighbors in the feature space

        ### Multiple Imputation: Creating multiple imputed datasets and combining them to obtain a final estimate

        ### Model-based imputation: Using machine learning models to predict missing values based on other available variables


# Example: Filling missing values with KNN imputation:
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Create a sample dataset with missing values
data = {'A': [1, 2, np.nan, 4, 5],
        'B': [6, np.nan, 8, 9, 10],
        'C': [11, 12, 13, np.nan, 15]}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Create a KNN imputer object with k=3
imputer = KNNImputer(n_neighbors=3)

# Fit the imputer on the DataFrame and transform the data
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("\nImputed DataFrame:")
print(df_imputed)
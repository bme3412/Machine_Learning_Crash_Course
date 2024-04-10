### Detecting outliers is an important step in data preprocessing and analysis. 

### Two commonly used techniques for detecting outliers are the Z-score method and the Interquartile Range (IQR) method

### Z-score Method: 

    ### The Z-score method, also known as the standard score, measures how many standard deviations a data point is from the mean of the dataset.
 
    ### It assumes that the data follows a normal distribution. 
        ### Here's how the Z-score is calculated: Z-score = (data point - mean) / standard deviation A data point is considered an outlier if its Z-score exceeds a certain threshold, typically 2.5 or 3 standard deviations from the mean. 

import numpy as np

def detect_outliers_z_score(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    outliers = data[np.abs(z_scores) > threshold]
    return outliers

# Example usage
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
outliers = detect_outliers_z_score(data)
print("Outliers (Z-score method):", outliers)

### Interquartile Range (IQR) Method: 
    ### The IQR method is a robust technique for detecting outliers that is less sensitive to extreme values compared to the Z-score method. It uses the quartiles of the dataset to identify outliers. Here's how the IQR method works:
    
    ### Calculate the first quartile (Q1) and the third quartile (Q3) of the dataset.
    ### Calculate the IQR as the difference between Q3 and Q1.
    ### Define the lower and upper bounds as:
        ### Lower bound = Q1 - 1.5 * IQR
        ### Upper bound = Q3 + 1.5 * IQR
        ### Any data point outside the lower and upper bounds is considered an outlier.

def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

# Example usage
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
outliers = detect_outliers_iqr(data)
print("Outliers (IQR method):", outliers)

### The Z-score method assumes a normal distribution and is sensitive to extreme values, while the IQR method is more robust to extreme values but may not work well for small datasets or datasets with a skewed distribution.
### Outliers are data points that lie far away from the central tendency of the data distribution. 
    ### They can be either unusually high or low values compared to the rest of the data. Outliers can occur due to various reasons, such as measurement errors, data entry mistakes, or genuine extreme values

    ### Outliers can have a significant impact on the performance and accuracy of machine learning models. Here are a few ways outliers can affect models: 
    
    ### a. Skewed Model Coefficients: Outliers can heavily influence the coefficients of linear models, such as linear regression. They can pull the regression line towards themselves, leading to inaccurate predictions for the majority of the data. 
    
    ### b. Biased Performance Metrics: Outliers can distort performance metrics like mean squared error (MSE) or mean absolute error (MAE). Since these metrics are sensitive to large errors, outliers can make the model appear worse than it actually is. 

    ### c. Overfitting: In some cases, outliers can cause the model to overfit, especially if the model is complex and tries to capture the outliers as part of the pattern. This can lead to poor generalization on unseen data.  ###

### Identifying Outliers: There are various techniques to identify outliers in a dataset. One common approach is to use statistical measures such as the interquartile range (IQR)

import numpy as np

def identify_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

# Example usage
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
outliers = identify_outliers(data)
print("Outliers:", outliers)

### Handling Outliers: Once outliers are identified, there are different strategies to handle them based on the specific problem and domain knowledge. 
    ## Some common approaches include: 
    
    ### a. Removal: If the outliers are due to data entry errors or are not representative of the underlying phenomenon, they can be removed from the dataset. However, this should be done with caution and justification. 
    
    ### b. Transformation: Outliers can be transformed using techniques like logarithmic transformation or square root transformation to reduce their impact on the model. 
     
    ### c. Robust Models: Some machine learning algorithms, such as decision trees and random forests, are less sensitive to outliers. These models can be used when outliers are present in the data. 
    
    ### d. Outlier Detection Models: Specialized models, such as the Isolation Forest or Local Outlier Factor (LOF), can be used to detect and handle outliers as part of the modeling process.


def remove_outliers(data, threshold=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return cleaned_data

# Example usage
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
cleaned_data = remove_outliers(data)
print("Cleaned Data:", cleaned_data)
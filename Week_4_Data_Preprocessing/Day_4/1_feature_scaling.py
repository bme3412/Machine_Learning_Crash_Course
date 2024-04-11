"""Min-Max Scaling (Normalization):
    Min-Max scaling, also known as normalization, is a technique that scales the features to a specific range, typically between 0 and 1 or -1 and 1.

    The formula for Min-Max scaling is: X_scaled = (X - X_min) / (X_max - X_min)

    Here, X is the original feature value, X_min is the minimum value of the feature, and X_max is the maximum value of the feature.

    After scaling, the minimum value of the feature becomes 0 (or -1), and the maximum value becomes 1.

    Min-Max scaling preserves the original distribution of the data and is useful when the data does not follow a Gaussian distribution.

    However, it is sensitive to outliers, as the presence of extreme values can compress the majority of the data points into a narrow range.

    Min-Max scaling is commonly used in image processing, where pixel intensities need to be scaled to a specific range."""

"""Standard Scaling (Standardization or Z-score Normalization):
    Standard scaling, also known as standardization or Z-score normalization, transforms the features to have zero mean and unit variance.

    The formula for Standard scaling is: X_scaled = (X - μ) / σ

    Here, X is the original feature value, μ is the mean of the feature, and σ is the standard deviation of the feature.

    After scaling, the mean of the feature becomes 0, and the standard deviation becomes 1.

    Standard scaling is useful when the data follows a Gaussian distribution or when the algorithm assumes that the features are normally distributed.

    It is less sensitive to outliers compared to Min-Max scaling, as the scaling is based on the mean and standard deviation rather than the minimum and maximum values.

    Standard scaling is commonly used in algorithms like Principal Component Analysis (PCA) and linear regression."""

"""Normalization (L1 and L2 Normalization):

    Normalization, in the context of feature scaling, refers to the process of scaling the features to have a unit norm.

    L1 Normalization (Manhattan Norm):

    L1 normalization scales the features so that the absolute values of the features sum up to 1.

    The formula for L1 normalization is: X_normalized = X / (|X1| + |X2| + ... + |Xn|)
    Here, X is the original feature vector, and |Xi| represents the absolute value of each feature.

    L1 normalization is useful when dealing with sparse data, as it can help emphasize the importance of non-zero features.

    L2 Normalization (Euclidean Norm):
    L2 normalization scales the features so that the squared values of the features sum up to 1.

    The formula for L2 normalization is: X_normalized = X / sqrt(X1^2 + X2^2 + ... + Xn^2)

    Here, X is the original feature vector, and Xi^2 represents the squared value of each feature.

    L2 normalization is commonly used in algorithms like Support Vector Machines (SVM) and K-Nearest Neighbors (KNN) to measure the distance between data points."""
"""Problem 1: Housing Prices
You are given a dataset of housing prices that includes various features such as the number of bedrooms, square footage, and location. Your task is to identify and handle outliers in the price column."""

    ### Load the dataset into a Pandas DataFrame.
    ### Calculate the Z-scores for the price column.
    ### Identify outliers based on a Z-score threshold of 3.
    ### Remove the outliers from the DataFrame.
    ### Print the number of outliers removed and the cleaned DataFrame.

"""Problem 2: Customer Spending
You have a dataset containing information about customer spending in an online store. The dataset includes the amount spent by each customer. Your goal is to detect and handle outliers in the spending data."""

    ## Load the dataset into a Pandas DataFrame.
    ## Use the IQR method to identify outliers in the spending column.
    ## Replace the outliers with the median value of the spending column.
    ## Print the number of outliers detected and the updated DataFrame.

"""Problem 3: Sensor Readings
You are working with a dataset of sensor readings from a manufacturing process. The dataset contains temperature measurements from various sensors. Your objective is to identify and handle outliers in the temperature data."""

    ### Load the dataset into a Pandas DataFrame.
    ### Use the Isolation Forest algorithm from Scikit-learn to detect outliers in the temperature column.
    ### Create a new DataFrame that excludes the outliers.
    ### Print the number of outliers detected and the cleaned DataFrame.

"""Problem 4: Student Exam Scores
You have a dataset of student exam scores for a particular subject. Your task is to identify and handle outliers in the exam score data."""

    ### Load the dataset into a Pandas DataFrame.
    ### Calculate the Z-scores for the exam score column.
    ### Identify outliers based on a Z-score threshold of 2.5.
    ### Winsorize the outliers by replacing them with the nearest non-outlier values.
    ### Print the number of outliers detected and the updated DataFrame.

"""Problem 5: Sales Data
You are analyzing sales data for a retail company. The dataset contains sales figures for different products. Your goal is to identify and handle outliers in the sales data."""    

    ### Load the dataset into a Pandas DataFrame.
    ## Use the Local Outlier Factor (LOF) algorithm from Scikit-learn to detect outliers in the sales column.
    ## Remove the outliers from the DataFrame.
    ## Print the number of outliers removed and the cleaned DataFrame.
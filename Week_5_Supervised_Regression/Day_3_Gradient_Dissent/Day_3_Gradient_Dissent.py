"""Day 3: Gradient Descent and Optimization"""

"""Learn about the gradient descent algorithm for optimization"""

### Gradient descent is an iterative optimization algorithm used to minimize a given cost function by adjusting the parameters of a model. 
    ### The goal is to find the set of parameters that minimizes the cost function, which measures the difference between the predicted output and the actual output.

    ### The algorithm works as follows:
        ### Initialize the parameters: Start by initializing the parameters of the model randomly or using some predefined values.

        ### Calculate the cost function: Evaluate the cost function using the current parameter values. The cost function quantifies the error or dissimilarity between the predicted output and the actual output.
        
        ## Compute the gradients: Calculate the gradients of the cost function with respect to each parameter. The gradient is a vector that points in the direction of the steepest ascent of the cost function.
        
        ## Update the parameters: Adjust the parameters in the opposite direction of the gradients, scaled by a learning rate. The learning rate determines the step size at which the parameters are updated. The update rule is typically:   

        ## Gradient descent is widely used in training various machine learning models, including linear regression, logistic regression, and neural networks. 
            #  It is a powerful optimization technique that enables models to learn from data and minimize the error between predictions and actual values.

        ## There are different variants of gradient descent:
            #  Batch Gradient Descent: Computes the gradients using the entire training dataset at each iteration. It can be computationally expensive for large datasets.
            
            # Stochastic Gradient Descent (SGD): Computes the gradients using a single randomly selected training example at each iteration. It is faster and can escape local minima but may exhibit noisy updates.
            
            # Mini-Batch Gradient Descent: Computes the gradients using a small batch of randomly selected training examples at each iteration. It strikes a balance between batch and stochastic gradient descent.

        ## The choice of the learning rate is crucial in gradient descent. 
            # A small learning rate leads to slow convergence, while a large learning rate may cause the algorithm to overshoot the minimum and diverge. 
            
            # Techniques like learning rate decay, adaptive learning rates (e.g., Adam, RMSprop), and momentum can help improve the convergence and stability of the algorithm.

"""implement gradient descent to find the optimal parameters (slope and intercept) that minimize the mean squared error (MSE) between the predicted and actual values."""

import numpy as np

# Generate sample data
X = np.array([1, 2, 3, 4, 5])  # Input features
y = np.array([3, 5, 7, 9, 11])  # Output values

# Initialize parameters
slope = 0.0
intercept = 0.0

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Training loop
for i in range(num_iterations):
    # Calculate predicted values
    y_pred = slope * X + intercept
    
    # Calculate gradients
    dSlope = np.mean(2 * (y_pred - y) * X)
    dIntercept = np.mean(2 * (y_pred - y))
    
    # Update parameters
    slope -= learning_rate * dSlope
    intercept -= learning_rate * dIntercept
    
    # Print the cost (MSE) every 100 iterations
    if i % 100 == 0:
        cost = np.mean((y_pred - y) ** 2)
        print(f"Iteration {i}: Cost = {cost:.4f}")

# Print the final parameters
print(f"\nFinal Parameters:")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")


"""Understand the concept of learning rate and its impact on convergence"""
    ### The learning rate is a hyperparameter in gradient descent that determines the step size at which the model's parameters are updated in the direction of the negative gradient. 
        
        # It controls how quickly or slowly the model learns from the training data.

        ## Small Learning Rate:
            # If the learning rate is set too small, the algorithm will take very small steps towards the minimum of the cost function.
            
            # This leads to slow convergence and may require a large number of iterations to reach the optimal solution.
            
            #  However, a small learning rate can be beneficial in ensuring a more precise convergence, especially when the algorithm is close to the minimum.
        
        ## Large Learning Rate:
            # If the learning rate is set too large, the algorithm takes large steps and may overshoot the minimum of the cost function.
            
            # This can cause the algorithm to diverge or oscillate around the minimum without converging.
            
            # In extreme cases, a large learning rate can lead to numerical instability and cause the cost function to increase instead of decrease.
        
        ## Optimal Learning Rate:
        
            # The optimal learning rate strikes a balance between convergence speed and precision.
            
            # It allows the algorithm to take reasonably sized steps towards the minimum while avoiding overshooting or divergence.
            
            #  The optimal learning rate depends on the specific problem, model architecture, and dataset.
        
        ## Learning Rate Schedules:
            # Instead of using a fixed learning rate throughout the training process, learning rate schedules can be employed.
            
            # Learning rate schedules adjust the learning rate dynamically during training based on a predefined strategy
            
            # Common learning rate schedules include:
                # Step Decay: The learning rate is reduced by a factor after a specified number of epochs or iterations.
                # Exponential Decay: The learning rate decays exponentially over time.
                # Cosine Annealing: The learning rate follows a cosine function, starting high and gradually decreasing before increasing again.
            
        ## Adaptive Learning Rates:
            # Adaptive learning rate algorithms, such as Adam, RMSprop, and Adagrad, automatically adjust the learning rate for each parameter based on its historical gradients.
            
            # These algorithms maintain a separate learning rate for each parameter and adapt them based on the magnitude of the gradients.
            
            # Adaptive learning rates can help accelerate convergence and handle sparse or noisy gradients effectively.


"""Implement gradient descent from scratch for a simple linear regression problem"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Initialize parameters
theta = np.random.randn(2, 1)

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Perform gradient descent
for iteration in range(num_iterations):
    # Compute the predicted values
    y_pred = np.dot(np.c_[np.ones((100, 1)), X], theta)
    
    # Compute the gradients
    gradients = (1/100) * np.dot(np.c_[np.ones((100, 1)), X].T, y_pred - y)
    
    # Update the parameters
    theta -= learning_rate * gradients
    
    # Compute the cost (Mean Squared Error)
    cost = (1/(2*100)) * np.sum((y_pred - y)**2)
    
    # Print the cost every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Cost = {cost:.4f}")

# Print the final parameters
print(f"\nFinal Parameters:")
print(f"Intercept: {theta[0][0]:.4f}")
print(f"Slope: {theta[1][0]:.4f}")

# Plot the data points and the fitted line
plt.scatter(X, y)
plt.plot(X, np.dot(np.c_[np.ones((100, 1)), X], theta), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.show()

"""Explore the different variations of gradient descent (batch, mini-batch, stochastic)"""

### The choice of the gradient descent variant depends on the specific problem, dataset size, and computational resources available. 
    ### Batch gradient descent is suitable for small datasets where the entire dataset can fit into memory. 
    
    ### Stochastic gradient descent is often used for large datasets or online learning scenarios where data arrives in a streaming fashion. 
    
    ## Mini-batch gradient descent is a popular choice for most machine learning tasks as it provides a good trade-off between stability and efficiency.
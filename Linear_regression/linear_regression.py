import numpy as np
import math
import random as rnd

class LinearRegression:
    # Function to read the training data from a CSV file and return X and y matrices
    @staticmethod
    def get_xmat_ymat(filepath):
        # Using np.genfromtxt to read the data directly into numpy arrays
        data = np.genfromtxt(filepath, delimiter=',')
        # First 7 columns are features (X matrix)
        t_x_mat = data[:, :7]
        # Last column is the target variable (y matrix)
        y_mat = data[:, 7].reshape(-1, 1)
        return t_x_mat.T, y_mat  # Return the transpose of X for easier matrix multiplication
    
    # Function to read the test data from a CSV file
    @staticmethod
    def get_test_xmat_ymat(filepath):
        # Reuse the same logic as the training data reader
        return LinearRegression.get_xmat_ymat(filepath)
    
    # Function to compute the optimal weight vector using the analytical solution (Normal Equation)
    @staticmethod
    def analytical_weight():
        # Get training data
        xmat, ymat = LinearRegression.get_xmat_ymat('./concrete/train.csv')
        # Solve the normal equation: w = (X^T * X)^(-1) * X^T * y
        # Using np.linalg.solve for better numerical stability than directly computing the inverse
        w = np.linalg.solve(xmat @ xmat.T, xmat @ ymat)
        # Write the weight vector to a file
        with open('./Data/analytical_weight', 'w') as f:
            f.write(str(w))
    
    # Function to compute the gradient for batch gradient descent
    @staticmethod
    def compute_grad(w, examples, y):
        # Compute predicted values: guesses = X^T * w
        guesses = examples.T @ w
        # Compute errors: errors = y - predictions
        errors = y - guesses
        # Compute the gradient: grad = -(X * errors) / number of samples
        grad = -(examples @ errors) / len(y)
        return grad
    
    # Function to compute the cost (mean squared error) for the given weight vector
    @staticmethod
    def get_cost(w, examples, y):
        # Compute predictions: guesses = X^T * w
        guesses = examples.T @ w
        # Compute errors: errors = y - guesses
        errors = y - guesses
        # Compute and return the cost (mean squared error divided by 2)
        return np.sum(errors**2) / (2 * len(y))
    
    # Function to update the weight vector using stochastic gradient descent
    @staticmethod
    def update_w_sto(w, example, y, r):
        # Compute error for a single sample: error = y - (x^T * w)
        error = y - (example.T @ w)
        # Update the weight vector: w_new = w + r * error * x
        new_w = w + r * error * example
        return new_w
    
    # Function to perform batch gradient descent
    @staticmethod
    def batch_grad_des(r):
        # Get training and test data
        xmat, ymat = LinearRegression.get_xmat_ymat('./concrete/train.csv')
        test_xmat, test_ymat = LinearRegression.get_test_xmat_ymat('./concrete/test.csv')
        
        # Initialize weight vector with zeros
        w_vec = np.zeros((7, 1))
        vec_diff = 1  # Difference between old and new weight vectors
        threshold = 1e-6  # Convergence threshold
        
        # Open file to record the cost at each step
        with open('./Data/batch_grad_cost', 'w') as f:
            # Continue until the weight vector converges
            while vec_diff > threshold:
                # Write the cost on the training data to file
                f.write(str(LinearRegression.get_cost(w_vec, xmat, ymat)) + "\n")
                
                # Compute the gradient
                grad_vec = LinearRegression.compute_grad(w_vec, xmat, ymat)
                # Update the weight vector
                new_w_vec = w_vec - r * grad_vec
                
                # Calculate the difference between the old and new weight vectors (for convergence check)
                vec_diff = np.linalg.norm(new_w_vec - w_vec)
                # Update the weight vector for the next iteration
                w_vec = new_w_vec
            
            # Write the final weight vector and test cost to the file
            f.write('w vector \n')
            f.write(str(w_vec))
            f.write('cost on test data\n' + str(LinearRegression.get_cost(w_vec, test_xmat, test_ymat)) + "\n")
    
    # Function to perform stochastic gradient descent
    @staticmethod
    def stochastic_grad_des(r):
        rnd.seed()  # Initialize the random number generator
        # Get training and test data
        xmat, ymat = LinearRegression.get_xmat_ymat('./concrete/train.csv')
        test_xmat, test_ymat = LinearRegression.get_xmat_ymat('./concrete/test.csv')
        
        # Initialize weight vector with zeros
        w_vec = np.zeros((7, 1))
        prev_cost = LinearRegression.get_cost(w_vec, xmat, ymat)  # Initial cost
        cost_diff = 1  # Difference between costs of consecutive iterations
        threshold = 1e-7  # Convergence threshold
        
        # Open file to record the cost at each step
        with open('./Data/stoch_grad_cost', 'w') as f:
            # Continue until the cost converges
            while cost_diff > threshold:
                # Write the current cost on the training data to file
                f.write(str(prev_cost) + "\n")
                
                # Select a random example for stochastic gradient descent
                i = rnd.randrange(0, len(ymat))
                # Update the weight vector using the selected random example
                new_w_vec = LinearRegression.update_w_sto(w_vec, xmat[:, i].reshape(-1, 1), ymat[i], r)
                
                # Compute the cost with the new weight vector
                current_cost = LinearRegression.get_cost(new_w_vec, xmat, ymat)
                # Compute the difference between the previous and current cost
                cost_diff = abs(prev_cost - current_cost)
                
                # Update previous cost and weight vector for the next iteration
                prev_cost = current_cost
                w_vec = new_w_vec
            
            # Write the final test cost and weight vector to the file
            f.write('test cost \n')
            f.write(str(LinearRegression.get_cost(w_vec, test_xmat, test_ymat)) + "\n")
            f.write('w vector \n')
            f.write(str(w_vec))

# Run the batch gradient descent and stochastic gradient descent with learning rate 0.01
LinearRegression.batch_grad_des(0.01)
LinearRegression.stochastic_grad_des(0.01)
LinearRegression.analytical_weight()

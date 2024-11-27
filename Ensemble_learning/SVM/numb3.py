import numpy as np
import pandas as pd
from scipy.optimize import minimize

class SVMDual:
    def __init__(self, C, kernel='linear', gamma=0.1):
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None

    def kernel(self, x1, x2):
        if self.kernel_type == 'linear':
            return np.dot(x1, x2)
        elif self.kernel_type == 'gaussian':
            return np.exp(-np.linalg.norm(x1 - x2)**2 / self.gamma)

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape

        # Compute the kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # Define the objective function
        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(alpha * y, K)) - np.sum(alpha)

        # Define constraints
        constraints = ({'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)},  # Sum(alpha_i * y_i) = 0
                       {'type': 'ineq', 'fun': lambda alpha: alpha},  # alpha_i >= 0
                       {'type': 'ineq', 'fun': lambda alpha: self.C - alpha})  # alpha_i <= C

        # Initial guess for alpha
        alpha0 = np.zeros(n_samples)

        # Solve the optimization problem
        solution = minimize(objective, alpha0, constraints=constraints, method='SLSQP')
        self.alpha = solution.x

        # Compute bias term
        support_vectors_idx = np.where((self.alpha > 1e-5) & (self.alpha < self.C))[0]
        self.b = np.mean([y[i] - np.sum(self.alpha * y * K[i]) for i in support_vectors_idx])

    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = np.sum(self.alpha * self.y * np.array([self.kernel(x, sv) for sv in self.X])) + self.b
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

    def compute_error(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred != y)

# Helper functions
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = np.where(y == 0, -1, 1)  # Convert labels to {1, -1}
    return X, y

def main():
    # Load datasets
    train_X, train_y = load_data("dataset/train.csv")
    test_X, test_y = load_data("dataset/test.csv")

    # Hyperparameters
    C_values = [100 / 873, 500 / 873, 700 / 873]
    gamma_values = [0.1, 0.5, 1, 5, 100]

    # Results storage
    results = []

    for C in C_values:
        for gamma in gamma_values:
            print(f"Running SVM with C = {C}, gamma = {gamma}")

            # Train SVM with Gaussian kernel
            model = SVMDual(C, kernel='gaussian', gamma=gamma)
            model.fit(train_X, train_y)

            # Compute errors
            train_error = model.compute_error(train_X, train_y)
            test_error = model.compute_error(test_X, test_y)
            print(f"Train error: {train_error}, Test error: {test_error}")

            results.append({
                "C": C,
                "gamma": gamma,
                "train_error": train_error,
                "test_error": test_error
            })

    # Output results
    print("\nResults:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()

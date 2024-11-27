import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class SVMPrimalSGD:
    def __init__(self, C, learning_rate_schedule, max_epochs=100):
        self.C = C
        self.learning_rate_schedule = learning_rate_schedule
        self.max_epochs = max_epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        
        for epoch in range(self.max_epochs):
            lr = self.learning_rate_schedule(epoch)
            # Shuffle data at the beginning of each epoch
            data = list(zip(X, y))
            random.shuffle(data)
            X, y = zip(*data)
            X = np.array(X)
            y = np.array(y)

            for i in range(n_samples):
                if y[i] * (np.dot(self.w, X[i]) + self.b) < 1:
                    self.w = self.w - lr * (self.w - self.C * y[i] * X[i])
                    self.b = self.b + lr * self.C * y[i]
                else:
                    self.w = self.w - lr * self.w

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def compute_error(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred != y)

# Helper functions
def learning_rate_schedule_1(t, gamma0=0.01, a=10):
    return gamma0 / (1 + (gamma0 / a) * t)

def learning_rate_schedule_2(t, gamma0=0.01):
    return gamma0 / (1 + t)

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
    max_epochs = 100

    # Results storage
    results = []

    for C in C_values:
        print(f"Running SVM with C = {C}")

        # Schedule 1
        print("Using learning rate schedule 1")
        model_1 = SVMPrimalSGD(C, lambda t: learning_rate_schedule_1(t, gamma0=0.01, a=10), max_epochs)
        model_1.fit(train_X, train_y)
        train_error_1 = model_1.compute_error(train_X, train_y)
        test_error_1 = model_1.compute_error(test_X, test_y)
        print(f"Train error: {train_error_1}, Test error: {test_error_1}")

        # Schedule 2
        print("Using learning rate schedule 2")
        model_2 = SVMPrimalSGD(C, lambda t: learning_rate_schedule_2(t, gamma0=0.01), max_epochs)
        model_2.fit(train_X, train_y)
        train_error_2 = model_2.compute_error(train_X, train_y)
        test_error_2 = model_2.compute_error(test_X, test_y)
        print(f"Train error: {train_error_2}, Test error: {test_error_2}")

        results.append({
            "C": C,
            "train_error_schedule_1": train_error_1,
            "test_error_schedule_1": test_error_1,
            "train_error_schedule_2": train_error_2,
            "test_error_schedule_2": test_error_2
        })

    # Output results
    print("\nResults:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()

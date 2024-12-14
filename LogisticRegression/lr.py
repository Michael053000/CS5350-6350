import numpy as np
import pandas as pd

# Load data without headers
train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

# Extract features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Neural network forward pass
def forward_pass(X, weights):
    a = X
    activations = [a]
    zs = []
    for w in weights:
        z = np.dot(a, w[:-1]) + w[-1]  # Include bias term
        zs.append(z)
        a = sigmoid(z)
        activations.append(a)
    return zs, activations

# Back-propagation
def back_propagation(X, y, zs, activations, weights):
    deltas = []
    delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
    deltas.append(delta)

    for l in range(len(weights) - 2, -1, -1):
        delta = np.dot(delta, weights[l + 1][:-1].T) * sigmoid_derivative(zs[l])
        deltas.append(delta)

    deltas.reverse()

    gradients = []
    for l in range(len(weights)):
        grad_w = np.dot(activations[l].T, deltas[l])
        grad_b = np.sum(deltas[l], axis=0, keepdims=True)
        gradients.append(np.vstack([grad_w, grad_b]))

    return gradients

# Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(X_train, y_train, hidden_layer_widths, gamma_0, d, num_epochs):
    layers = [X_train.shape[1]] + hidden_layer_widths + [1]
    weights = [np.random.randn(layers[i] + 1, layers[i + 1]) for i in range(len(layers) - 1)]

    errors = []
    for epoch in range(num_epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for t, (x, y) in enumerate(zip(X_train_shuffled, y_train_shuffled)):
            gamma_t = gamma_0 / (1 + (gamma_0 / d) * t)
            zs, activations = forward_pass(x.reshape(1, -1), weights)
            gradients = back_propagation(x.reshape(1, -1), y, zs, activations, weights)
            for i in range(len(weights)):
                weights[i] -= gamma_t * gradients[i]

        # Calculate training error
        _, activations = forward_pass(X_train, weights)
        y_pred_train = (activations[-1] > 0.5).astype(int)
        train_error = np.mean(y_pred_train != y_train)
        errors.append(train_error)

    return weights, errors

# Evaluation
def evaluate(weights, X, y):
    _, activations = forward_pass(X, weights)
    y_pred = (activations[-1] > 0.5).astype(int)
    error = np.mean(y_pred != y)
    return error

# Experiment settings
hidden_layer_widths = [5, 10, 25, 50, 100]
gamma_0 = 0.1
d = 0.01
num_epochs = 50

results = []
for width in hidden_layer_widths:
    print(f"\nTraining with hidden layer width: {width}")
    weights, errors = stochastic_gradient_descent(X_train, y_train, [width, width], gamma_0, d, num_epochs)

    train_error = evaluate(weights, X_train, y_train)
    test_error = evaluate(weights, X_test, y_test)

    print(f"Final Training Error: {train_error:.4f}")
    print(f"Final Test Error: {test_error:.4f}")

    results.append((width, train_error, test_error))

# Print summary
print("\nSummary of Results:")
print("Width\tTrain Error\tTest Error")
for width, train_error, test_error in results:
    print(f"{width}\t{train_error:.4f}\t{test_error:.4f}")

# Analysis for zero-initialized weights
print("\nTraining with zero-initialized weights")
weights = [np.zeros((X_train.shape[1] + 1, width)) for width in hidden_layer_widths]
weights, errors = stochastic_gradient_descent(X_train, y_train, [10, 10], gamma_0, d, num_epochs)
train_error = evaluate(weights, X_train, y_train)
test_error = evaluate(weights, X_test, y_test)
print(f"Final Training Error with zero initialization: {train_error:.4f}")
print(f"Final Test Error with zero initialization: {test_error:.4f}")

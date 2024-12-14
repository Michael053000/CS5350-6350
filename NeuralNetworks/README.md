# Neural Network Implementation (NN)

## Description
This script implements a three-layer artificial neural network for classification using stochastic gradient descent (SGD) and the back-propagation algorithm. Variations include experiments with random weight initialization, zero initialization, and PyTorch-based implementation with advanced features.

## Features
### 1. `Sgradient`
- Implements SGD for training the network with random weight initialization.
- Uses a learning rate schedule:
  \[
  \gamma_t = \frac{\gamma_0}{1 + \frac{\gamma_0}{d} t}
  \]
- Supports hidden layer widths: `{5, 10, 25, 50, 100}`.
- Reports training and test error for each width.
- Generates a convergence plot of the loss over epochs.

### 2. `SgradientZero`
- Same as `Sgradient.py`, but initializes all weights to zero to demonstrate symmetry-breaking issues.
- Reports training and test error for each width.
- Generates a convergence plot of the loss over epochs.

### 3. `BackPropagation`
- Implements back-propagation to compute gradients and update weights for the network.
- Uses random Gaussian initialization for weights.
- Supports varying hidden layer widths: `{5, 10, 25, 50, 100}`.
- Reports training and test error for each width.
- Generates a convergence plot of the loss over epochs.

### 4. `pytorch`
- PyTorch-based implementation with support for varying depth, width, and activation functions.
- Supports activation functions: `tanh` (with Xavier initialization) and `ReLU` (with He initialization).
- Depth: `{3, 5, 9}`.
- Width: `{5, 10, 25, 50, 100}`.
- Uses Adam optimizer for training.
- Reports training and test error for each depth-width combination.
- Generates a convergence plot of the loss over epochs.

## How to Run
1. Ensure the dataset is placed in the directory with the following files:
   - `train.csv`
   - `test.csv`

2. Install the required Python libraries:
   ```bash
   pip install numpy pandas matplotlib torch scikit-learn tensorflow

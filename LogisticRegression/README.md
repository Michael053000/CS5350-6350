---


### README for Logistic Regression (LR)

markdown
# Logistic Regression Implementation (LR)

## Description
This script implements logistic regression for binary classification using stochastic gradient descent (SGD). Two variations are implemented:
1. **Maximum A Posteriori (MAP) Estimation** with Gaussian priors.
2. **Maximum Likelihood (ML) Estimation** without priors.

## Features
### 1. MAP Estimation
- Includes Gaussian prior with variance $v$:
  \[
  p(w_i) = \mathcal{N}(w_i | 0, v)
  \]
- Objective function:
  \[
  J(w) = -\sum_{i=1}^N \log \sigma(y_i w^T x_i) + \frac{1}{2v} \|w\|^2
  \]
- Gradient:
  \[
  \nabla J(w) = -\sum_{i=1}^N y_i x_i (1 - \sigma(y_i w^T x_i)) + \frac{1}{v} w
  \]
- Variance $v$ is chosen from $\{0.01, 0.1, 0.5, 1, 3, 5, 10, 100\}$.
- Uses a learning rate schedule:
  \[
  \gamma_t = \frac{\gamma_0}{1 + \frac{\gamma_0}{d} t}
  \]
- Reports training and test errors for each variance setting.
- Generates convergence plots of the objective function.

### 2. ML Estimation
- Maximizes the logistic likelihood of the data without prior regularization.
- Objective function:
  \[
  J(w) = -\sum_{i=1}^N \log \sigma(y_i w^T x_i)
  \]
- Gradient:
  \[
  \nabla J(w) = -\sum_{i=1}^N y_i x_i (1 - \sigma(y_i w^T x_i))
  \]
- Uses the same learning rate schedule.
- Reports training and test errors for different learning rates.
- Generates convergence plots of the objective function.

## How to Run
1. Ensure the dataset is placed in the `classification/` directory with the following files:
   - `classification/train.csv`
   - `classification/test.csv`

2. Install the required Python libraries:
   ```bash
   pip install numpy pandas matplotlib

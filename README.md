
# Machine Learning Library

## Decision Tree Implementation

This is a basic implementation of a Decision Tree algorithm. You can use this code to train a decision tree on any dataset.

### How to use

#### Learning a Decision Tree
To train a decision tree on your dataset, use the following command:

```python
from DecisionTree import DecisionTree

# Example: Training a decision tree
dt = DecisionTree(criterion='entropy', max_depth=5)
dt.fit(X_train, y_train)

# Making predictions
predictions = dt.predict(X_test)
```

### Parameters
- `criterion`: The function to measure the quality of a split. Supported criteria are:
  - `gini`: for Gini impurity.
  - `entropy`: for information gain.
- `max_depth`: The maximum depth of the tree. Default is `None`, meaning the tree can grow until all leaves are pure.

### Example Usage
To run the decision tree with the default parameters:
```python
dt = DecisionTree()
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
```

## Folder Structure

- `DecisionTree`: Contains the implementation of the decision tree algorithm.
- `EnsembleLearning`: Placeholder for ensemble learning methods (e.g., Bagging, Boosting).
- `LinearRegression`: Placeholder for linear regression implementations.

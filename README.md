This is a machine learning library developed by Hosoo Lee for CS 5350 in University of Utah

# Decision Tree Implementation

## How to use

### Learning a Decision Tree
To train a decision tree on your dataset, you can use the following command:

```python
from DecisionTree import DecisionTree

# Example: Training a decision tree
dt = DecisionTree(criterion='entropy', max_depth=5)
dt.fit(X_train, y_train)

# Making predictions
predictions = dt.predict(X_test)
```

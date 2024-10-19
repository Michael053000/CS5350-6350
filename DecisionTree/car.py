import pandas as pd
import numpy as np
from collections import Counter

# This class defines each node of the decision tree.
# It contains an attribute to split on, a label if it's a leaf, and a dictionary of child nodes.
class TreeNode:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute  # Attribute used to split data at this node
        self.label = label  # Class label if this node is a leaf node
        self.children = {}  # Dictionary to store child nodes for each attribute value

# Function to calculate entropy of a dataset
# Entropy is a measure of uncertainty in the data
def calculate_entropy(labels):
    values, counts = np.unique(labels, return_counts=True)  # Get unique labels and their counts
    probabilities = counts / len(labels)  # Calculate probabilities of each class
    # Compute entropy, adding a small epsilon to avoid log(0)
    return -np.sum([p * np.log2(p + 1e-9) for p in probabilities])

# Function to calculate Gini index, another impurity measure
# Gini index reflects the probability of misclassification
def calculate_gini(labels):
    values, counts = np.unique(labels, return_counts=True)  # Get unique labels and their counts
    probabilities = counts / len(labels)  # Calculate probabilities of each class
    # Gini is 1 - sum of squared probabilities
    return 1 - np.sum([p ** 2 for p in probabilities])

# Function to calculate the majority error
# Majority error simply measures the proportion of instances not belonging to the majority class
def majority_error(labels):
    most_common = Counter(labels).most_common(1)[0][1]  # Get the count of the most common class
    return 1 - (most_common / len(labels))  # Return the proportion of the non-majority classes

# Function to calculate information gain for a particular attribute
# Information gain is the reduction in uncertainty (entropy, Gini, or majority error) after splitting on an attribute
def gain_information(data, labels, attribute, measure_func):
    total_impurity = measure_func(labels)  # Measure impurity before splitting
    values = data[attribute].unique()  # Get unique values of the attribute to split on
    weighted_impurity = 0
    # For each value of the attribute, calculate weighted impurity of the subset
    for value in values:
        subset = labels[data[attribute] == value]  # Subset labels where attribute equals the value
        weighted_impurity += (len(subset) / len(labels)) * measure_func(subset)  # Weighted impurity
    # Information gain is the reduction in impurity
    return total_impurity - weighted_impurity

# Recursive function to build a decision tree using ID3 algorithm
# It will split based on the attribute that gives the highest information gain
def build_decision_tree(data, labels, available_attributes, max_depth, measure_func, depth=0):
    # Base cases: If all labels are the same, return a leaf node with the label
    if len(np.unique(labels)) == 1:
        return TreeNode(label=labels.iloc[0])
    # If we have no more attributes to split on or reached max depth, return the majority class
    if len(available_attributes) == 0 or depth == max_depth:
        common_label = Counter(labels).most_common(1)[0][0]
        return TreeNode(label=common_label)

    # Find the best attribute to split on by maximizing information gain
    best_gain = -1
    best_attribute = None
    for attribute in available_attributes:
        gain = gain_information(data, labels, attribute, measure_func)  # Compute gain for the attribute
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute

    # If no attribute improves information gain, return majority class
    if best_attribute is None:
        return TreeNode(label=Counter(labels).most_common(1)[0][0])

    # Create a new decision node for the best attribute
    node = TreeNode(attribute=best_attribute)
    unique_values = data[best_attribute].unique()  # Get unique values for the attribute

    # Recursively build the tree for each subset of data where the attribute has a specific value
    for value in unique_values:
        subset_data = data[data[best_attribute] == value]
        subset_labels = labels[data[best_attribute] == value]
        # Exclude the current attribute and move to the next depth level
        remaining_attributes = [a for a in available_attributes if a != best_attribute]
        node.children[value] = build_decision_tree(subset_data, subset_labels, remaining_attributes, max_depth, measure_func, depth + 1)

    return node

# Function to make a prediction by traversing the decision tree
# The function will move down the tree following the attribute values in the instance
def classify(tree, instance):
    # If we reached a leaf node, return its label
    if tree.label is not None:
        return tree.label
    attr_value = instance[tree.attribute]  # Get the value of the attribute at the current node
    # Traverse the tree based on the attribute value
    if attr_value in tree.children:
        return classify(tree.children[attr_value], instance)  # Recurse down the tree
    else:
        # If the attribute value is not found in the children, return the majority class of the children
        return Counter([child.label for child in tree.children.values() if child.label is not None]).most_common(1)[0][0]

# Function to calculate the accuracy of the decision tree on a dataset
# Accuracy is the percentage of correct predictions
def calculate_accuracy(tree, data, labels):
    predictions = data.apply(lambda row: classify(tree, row), axis=1)  # Make predictions for each instance
    return (predictions == labels).mean()  # Return the accuracy (correct predictions / total)

# Load and preprocess the dataset
train_data = pd.read_csv('/car/train.csv', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
test_data = pd.read_csv('/car/test.csv', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

# Separate features (X) and labels (y) for both training and testing datasets
X_train, y_train = train_data.drop(columns='class'), train_data['class']
X_test, y_test = test_data.drop(columns='class'), test_data['class']

# Define the impurity measures (entropy, Gini index, majority error)
criteria = {
    'Entropy': calculate_entropy,
    'Gini': calculate_gini,
    'Majority Error': majority_error
}

# Store the results for evaluation
results = []

# Loop through each impurity measure and each maximum tree depth (1 to 6)
for criterion, func in criteria.items():
    for depth in range(1, 7):
        # Build a decision tree using the current impurity measure and depth
        tree = build_decision_tree(X_train, y_train, X_train.columns, depth, func)
        # Calculate accuracy on training and testing datasets
        train_accuracy = calculate_accuracy(tree, X_train, y_train)
        test_accuracy = calculate_accuracy(tree, X_test, y_test)
        # Append results for reporting
        results.append({
            'Criterion': criterion,
            'Max Depth': depth,
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy
        })

# Convert results to a DataFrame and print them for analysis
results_df = pd.DataFrame(results)
print(results_df)


# Step 1: Define a Decision Stump (a shallow decision tree with max depth = 1)
class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1  # Direction of inequality
        self.alpha = None  # Weight of this stump in AdaBoost

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_gain = -float('inf')

        # Try all features and thresholds
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                gain, polarity = self._calculate_weighted_gain(feature_values, y, threshold, weights)
                if gain > best_gain:
                    best_gain = gain
                    self.feature = feature_idx
                    self.threshold = threshold
                    self.polarity = polarity

    def predict(self, X):
        feature_values = X[:, self.feature]
        predictions = np.ones(len(X))
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1
        return predictions

    def _calculate_weighted_gain(self, feature_values, y, threshold, weights):
        left_mask = feature_values < threshold
        right_mask = feature_values >= threshold
        left_weight = np.sum(weights[left_mask])
        right_weight = np.sum(weights[right_mask])

        left_majority = np.sign(np.dot(weights[left_mask], y[left_mask]))
        right_majority = np.sign(np.dot(weights[right_mask], y[right_mask]))

        left_error = np.sum(weights[left_mask] * (y[left_mask] != left_majority))
        right_error = np.sum(weights[right_mask] * (y[right_mask] != right_majority))

        weighted_error = left_error + right_error
        return 1 - weighted_error, 1 if left_majority == 1 else -1


# Step 2: Implement AdaBoost
class AdaBoost:
    def __init__(self, n_estimators=500):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)

            predictions = stump.predict(X)

            error = np.dot(weights, predictions != y) / np.sum(weights)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        stump_predictions = [stump.predict(X) for stump in self.stumps]
        final_predictions = np.sum([alpha * pred for alpha, pred in zip(self.alphas, stump_predictions)], axis=0)
        return np.sign(final_predictions)


# Load your car dataset and apply AdaBoost
car_train_data = pd.read_csv('/car/train.csv', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
car_test_data = pd.read_csv('/car/test.csv', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

X_train = car_train_data.drop(columns='class').values
y_train = car_train_data['class'].replace({'unacc': -1, 'acc': 1}).values
X_test = car_test_data.drop(columns='class').values
y_test = car_test_data['class'].replace({'unacc': -1, 'acc': 1}).values

# Train AdaBoost
adaboost = AdaBoost(n_estimators=500)
adaboost.fit(X_train, y_train)

# Predict using AdaBoost
train_predictions = adaboost.predict(X_train)
test_predictions = adaboost.predict(X_test)

train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
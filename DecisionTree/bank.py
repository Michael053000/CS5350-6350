
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# A class to represent a decision tree node
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, outcome=None):
        self.feature = feature  # Feature to be split
        self.threshold = threshold  # Threshold for numerical splits
        self.outcome = outcome  # Class label for leaf nodes
        self.children = {}  # Child nodes dictionary

# Function to calculate entropy
def calculate_entropy(labels):
    total_count = len(labels)
    class_counts = Counter(labels)
    entropy_value = sum((-count / total_count) * np.log2(count / total_count + 1e-9) for count in class_counts.values())
    return entropy_value

# Function to calculate the Gini impurity score
def calculate_gini(labels):
    total_count = len(labels)
    class_counts = Counter(labels)
    gini_value = 1 - sum((count / total_count) ** 2 for count in class_counts.values())
    return gini_value

# Function to calculate the majority error
def calculate_majority_error(labels):
    total_count = len(labels)
    most_common_class_count = max(Counter(labels).values())
    majority_error_value = 1 - (most_common_class_count / total_count)
    return majority_error_value

# Compute the information gain of splitting by a particular feature
def compute_information_gain(data, labels, split_feature, impurity_fn):
    total_impurity = impurity_fn(labels)

    unique_values = data[split_feature].unique()
    weighted_impurity_sum = 0
    for value in unique_values:
        subset = labels[data[split_feature] == value]
        weighted_impurity_sum += (len(subset) / len(labels)) * impurity_fn(subset)

    return total_impurity - weighted_impurity_sum

# Main function to build the decision tree using ID3
def construct_decision_tree(data, labels, features, depth_limit, impurity_fn, current_depth=0):
    if len(set(labels)) == 1:  # If all labels are the same
        return DecisionTreeNode(outcome=labels.iloc[0])

    if not features or current_depth == depth_limit:
        most_common_class = labels.mode()[0]
        return DecisionTreeNode(outcome=most_common_class)

    # Calculate information gain for all features and choose the best one to split on
    gains = {feature: compute_information_gain(data, labels, feature, impurity_fn) for feature in features}
    best_feature = max(gains, key=gains.get)

    node = DecisionTreeNode(feature=best_feature)

    for value in data[best_feature].unique():
        subset_data = data[data[best_feature] == value]
        subset_labels = labels[data[best_feature] == value]
        remaining_features = [f for f in features if f != best_feature]
        node.children[value] = construct_decision_tree(subset_data, subset_labels, remaining_features, depth_limit, impurity_fn, current_depth + 1)

    return node

# Function to classify a single example using the decision tree
def classify_with_tree(node, instance):
    if node.outcome is not None:
        return node.outcome  # If it's a leaf, return the outcome

    feature_value = instance[node.feature]
    if feature_value in node.children:
        return classify_with_tree(node.children[feature_value], instance)
    else:
        return None  # If value not found in the tree, return None

# Function to calculate accuracy of the decision tree on a dataset
def calculate_accuracy(tree, dataset, labels):
    predictions = [classify_with_tree(tree, dataset.iloc[i]) for i in range(len(dataset))]
    return np.mean(np.array(predictions) == np.array(labels))

# Bank Dataset: Load data directly from CSV URLs instead of local files
columns = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
    'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
    'previous', 'poutcome', 'y'
]

# Load the train and test data (if hosted online, replace with the actual URL)
bank_data = pd.read_csv('/bank/train.csv', names=columns)
bank_test_data = pd.read_csv('/bank/test.csv', names=columns)

# Binary conversion of numerical columns based on their median value
binary_columns = ['age', 'balance', 'day', 'duration', 'pdays', 'previous', 'campaign']
for column_name in binary_columns:
    bank_data[column_name] = bank_data[column_name] > bank_data[column_name].median()
    bank_test_data[column_name] = bank_test_data[column_name] > bank_data[column_name].median()  # Use train data's median

# Prepare the training and test data for the decision tree algorithm
X_train = bank_data.drop('y', axis=1)
y_train = bank_data['y']
X_test = bank_test_data.drop('y', axis=1)
y_test = bank_test_data['y']

# Impurity functions and depth configurations
impurity_functions = {
    'Entropy': calculate_entropy,
    'Gini': calculate_gini,
    'Majority Error': calculate_majority_error
}

max_depths = range(1, 7)  # You can increase the depth as needed

# Store results for analysis
evaluation_results = []

# Build and evaluate the decision tree for each impurity measure and depth limit
for impurity_name, impurity_fn in impurity_functions.items():
    for depth in max_depths:
        tree_model = construct_decision_tree(X_train, y_train, X_train.columns.tolist(), depth, impurity_fn)
        train_accuracy = calculate_accuracy(tree_model, X_train, y_train)
        test_accuracy = calculate_accuracy(tree_model, X_test, y_test)
        evaluation_results.append({
            'Impurity': impurity_name,
            'Depth': depth,
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy
        })

# Convert evaluation results into a DataFrame for easy viewing
evaluation_df = pd.DataFrame(evaluation_results)
evaluation_df['Train Error'] = 1 - evaluation_df['Train Accuracy']
evaluation_df['Test Error'] = 1 - evaluation_df['Test Accuracy']

# Print the results as a table
print("\nError Table for Different Impurity Functions and Depths:")
error_table = evaluation_df.pivot_table(values=['Train Error', 'Test Error'], index=['Impurity', 'Depth'])
print(error_table.to_string(float_format="{:.4f}".format))

# Calculate and display the average error across all depths for each impurity function
avg_errors = evaluation_df.groupby('Impurity')[['Train Error', 'Test Error']].mean()
print("\nAverage Errors Across All Depths:")
print(avg_errors.to_string(float_format="{:.4f}".format))

import math
import random

# Node class for the Decision Tree
class Node:
    def __init__(self, data):
        self.data = data  # Holds attribute or class label
        self.mcl = None  # Most common label
        self.children = {}  # Children nodes in the tree

# Decision Tree class with methods for constructing the tree and helper functions
class DecTree:
    # Returns True if the attribute is numerical (specific to the bank dataset)
    @staticmethod
    def is_numerical(attribute):
        num_attributes = [0, 5, 9, 11, 12, 13, 14]  # Example of numerical attributes
        return attribute in num_attributes

    # Reads data from a CSV file and returns a list of dictionaries (records) and the label index
    @staticmethod
    def read_file(filepath):
        data_array = []
        with open(filepath, 'r') as datafile:
            for line in datafile:
                values = line.strip().split(',')
                record = {i: value for i, value in enumerate(values)}
                data_array.append(record)
        return data_array, len(data_array[0]) - 1  # Return data and the label index

    # Calculates the median value for a specific attribute in the data
    @staticmethod
    def get_median(data, variable):
        values = [float(record[variable]) for record in data]
        values.sort()
        return values[len(values) // 2]  # Return the median value

    # Returns a list of attributes for the given data
    @staticmethod
    def make_attributes_array(num_attributes):
        return list(range(num_attributes))  # Return a list of attribute indices

    # Returns a subset of the data where the variable equals a specific value
    @staticmethod
    def data_subset(data, variable, val):
        return [record for record in data if record[variable] == val]

    # Creates a dictionary of unique values for a variable and their frequencies
    @staticmethod
    def make_variable_dict(data, variable):
        var_dict = {}
        for record in data:
            value = record[variable]
            if value in var_dict:
                var_dict[value] += 1
            else:
                var_dict[value] = 1
        return var_dict

    # Returns the most common value in the provided dictionary (used for classification)
    @staticmethod
    def get_most_common(var_dict):
        return max(var_dict, key=var_dict.get)  # Return the key with the highest frequency

    # Checks if all records in the data have the same label (i.e., pure node)
    @staticmethod
    def label_check(data, label):
        first_label = data[0][label]
        return all(record[label] == first_label for record in data)

    # Calculates the entropy of a variable based on its frequency distribution
    @staticmethod
    def get_entropy(size, var_dict):
        entropy = 0
        for key in var_dict:
            prob = var_dict[key] / size
            entropy -= prob * math.log2(prob)  # Standard entropy calculation
        return entropy

    # Calculates the majority error (misclassification rate) for a given variable
    @staticmethod
    def get_majority_error(size, var_dict):
        if size == 0:
            return 0
        most_common_value = DecTree.get_most_common(var_dict)
        return (size - var_dict[most_common_value]) / size

    # Calculates the Gini index for a given variable
    @staticmethod
    def get_gini(size, var_dict):
        gini = 1
        for key in var_dict:
            prob = var_dict[key] / size
            gini -= prob ** 2  # Gini impurity calculation
        return gini

    # Finds the attribute with the highest information gain based on entropy
    @staticmethod
    def find_best_gain_e(data, attributes, label):
        label_dict = DecTree.make_variable_dict(data, label)
        set_entropy = DecTree.get_entropy(len(data), label_dict)
        max_gain = 0
        best_attribute = attributes[0]
        for attribute in attributes:
            gain = set_entropy
            var_dict = DecTree.make_variable_dict(data, attribute)
            for key in var_dict:
                subset = DecTree.data_subset(data, attribute, key)
                subset_dict = DecTree.make_variable_dict(subset, label)
                subset_entropy = DecTree.get_entropy(len(subset), subset_dict)
                gain -= (len(subset) / len(data)) * subset_entropy
            if gain > max_gain:
                max_gain = gain
                best_attribute = attribute
        return best_attribute

    # Builds the decision tree using data, attributes, and specific splitting method (entropy, majority error, or Gini index)
    @staticmethod
    def build_tree(data, attributes, label, node, depth, split):
        node.mcl = DecTree.get_most_common(DecTree.make_variable_dict(data, label))  # Set most common label
        if depth == 1 or not attributes:  # Base case: return a leaf node
            node.data = node.mcl
            return node
        elif DecTree.label_check(data, label):  # Base case: pure node (same label for all data)
            node.data = data[0][label]
            return node
        else:
            # Determine the best attribute based on the chosen splitting method
            if split == 1:
                best_gain = DecTree.find_best_gain_e(data, attributes, label)
            elif split == 2:
                best_gain = DecTree.find_best_gain_me(data, attributes, label)
            else:
                best_gain = DecTree.find_best_gain_g(data, attributes, label)

            node.data = best_gain  # Set node to the best attribute
            var_dict = DecTree.make_variable_dict(data, best_gain)
            for key in var_dict:
                subset = DecTree.data_subset(data, best_gain, key)
                child_node = Node(None)
                if not subset:
                    child_node.data = node.mcl
                else:
                    attributes.remove(best_gain)
                    DecTree.build_tree(subset, attributes, label, child_node, depth - 1, split)
                    attributes.append(best_gain)
                node.children[key] = child_node  # Add the child node to the tree
            return node

# Adaboost implementation with tree stumps (shallow trees)
class Adaboost:
    # Calculates the error for a tree and updates weights
    @staticmethod
    def err_and_update(data, tree, label):
        error_set, correct_set = [], []
        error_weight = 0
        for record in data:
            if DecTree.predict_binary_label(tree, record) == record[label]:
                correct_set.append(record)
            else:
                error_set.append(record)
                error_weight += record['w']
        change = 0.5 * math.log((1 - error_weight) / error_weight) if error_weight > 0 else 0
        err_val, cor_val = math.exp(change), math.exp(-change)
        for record in error_set:
            record['w'] *= err_val
        for record in correct_set:
            record['w'] *= cor_val
        # Normalize weights
        total_weight = sum(record['w'] for record in data)
        for record in data:
            record['w'] /= total_weight
        return change

    # Adaboost algorithm: trains multiple stumps and writes errors to a file
    @staticmethod
    def adaBoost(train_data, test_data, label):
        attributes = DecTree.make_attributes_array(label)
        for record in train_data:
            record['w'] = 1 / len(train_data)  # Initialize weights
        forest, votes = [], []
        with open("./Data/AdaBoost_Data.txt", "w") as f:
            for i in range(500):  # 500 stumps
                tree = Adaboost.build_stump(train_data, attributes, label)
                alpha = Adaboost.err_and_update(train_data, tree, label)
                forest.append(tree)
                votes.append(alpha)
                train_error = Adaboost.find_error(train_data, forest, label, votes)
                test_error = Adaboost.find_error(test_data, forest, label, votes)
                f.write(f"T: {i} train error: {train_error} test error: {test_error}\n")

# Bagging implementation: trains multiple decision trees on random subsets
class Bagging:
    # Generates a random sample with replacement (bootstrap sample)
    @staticmethod
    def randData(data):
        return [random.choice(data) for _ in range(len(data))]

    # Computes the bagged prediction for binary classification
    @staticmethod
    def bag_prediction(trees, record):
        yes_count = sum(1 for tree in trees if DecTree.predict_binary_label(tree, record) == 'yes')
        return 1 if yes_count >= len(trees) / 2 else -1

    # Calculates the training and testing errors for bagged trees
    @staticmethod
    def BaggingErrors(trees, train_data, test_data, label):
        train_error = sum(Bagging.bagPredictionErr(trees, record, label) for record in train_data) / len(train_data)
        test_error = sum(Bagging.bagPredictionErr(trees, record, label) for record in test_data) / len(test_data)
        return train_error, test_error

# Method class to run the various algorithms and save output
class methods:
    # Stump-based Adaboost
    @staticmethod
    def stump_boost():
        train_data = DecTree.read_file('./bank/train.csv')
        test_data = DecTree.read_file('./bank/test.csv')[0]
        Adaboost.adaBoost(train_data[0], test_data, train_data[1])

    # Bagging method: trains multiple trees and writes the error rates
    @staticmethod
    def bagging():
        train_data = DecTree.read_file('./bank/train.csv')
        test_data = DecTree.read_file('./bank/test.csv')
        attributes = DecTree.make_attributes_array(train_data[1])
        tree_array = []
        with open("./Data/Bagging_Data.txt", "w") as f:
            for i in range(1, 101):
                curr_data = Bagging.randData(train_data[0])
                tree_array.append(DecTree.build_binary_tree(curr_data, attributes, train_data[1], Node(None), 100, 1))
                errors = Bagging.BaggingErrors(tree_array, train_data[0], test_data[0], train_data[1])
                f.write(f"{i} {errors[0]} {errors[1]}\n")

methods.stump_boost()
methods.bagging()

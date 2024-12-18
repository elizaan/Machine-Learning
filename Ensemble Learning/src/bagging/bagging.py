import numpy as np
import matplotlib.pyplot as plt
from decision_tree.decision_tree import ID3, predict
from decision_tree.feature_def import get_feature_definitions

class BaggedTrees:
    def __init__(self, num_trees, max_depth):
        """
        Initialize the Bagged Trees model.
        Args:
            num_trees (int): The number of decision trees to use in the ensemble.
            max_depth (int): The maximum depth for each individual tree.
        """
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []


    def fit(self, train_data, Feature, Column, Numeric_Attributes, info_gain):
        """
        Train the Bagged Trees model using the given training data.
        Args:
            train_data (np.array): Training data.
            Feature (dict): Feature definitions.
            Column (list): List of column names.
            Numeric_Attributes (list): List of numeric attributes.
            info_gain (str): Criterion for splitting ('entropy', 'gini_index', or 'majority_error').
        """
        n_samples = len(train_data)
        self.trees = []

        for i in range(self.num_trees):
            # Create a bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_data = train_data[bootstrap_indices]

            # Debugging: Check the diversity of the bootstrap sample
            unique, counts = np.unique(bootstrap_data[:, -1], return_counts=True)
            # print(f"Tree {i+1} - Bootstrap Sample Label Distribution: {dict(zip(unique, counts))}")

            # Train a new decision tree using the bootstrap sample
            S_idx = np.arange(n_samples)
            attributes = Column.copy()
            tree = ID3(S_idx, attributes, 1, self.max_depth, bootstrap_data, info_gain, Column, Numeric_Attributes, Feature)

            # Save the trained tree
            self.trees.append(tree)


    def bagged_predict(self, data, Column, Numeric_Attributes):
        """
        Predict the label for a given instance using the Bagged Trees model.
        Args:
            data (np.array): Array representing a single instance's features.
            Column (list): List of column names.
            Numeric_Attributes (list): List of numeric attributes.
        Returns:
            int: Predicted label (majority vote from all trees).
        """
        # Collect predictions from each tree
        tree_predictions = [predict(data, tree, Column, Numeric_Attributes) for tree in self.trees]
        
        # Debugging: Print predictions from each tree
        # print(f"Tree Predictions: {tree_predictions}")

        # Perform majority voting
        majority_vote = max(set(tree_predictions), key=tree_predictions.count)
        
        # Debugging: Print the final aggregated prediction
        # print(f"Aggregated Prediction: {majority_vote}")
        
        return majority_vote


    # def score(self, data, true_labels, Column, Numeric_Attributes):
    #     """
    #     Calculate the accuracy of the Bagged Trees model.
    #     Args:
    #         data (np.array): Data to calculate accuracy for.
    #         true_labels (np.array): True labels.
    #         Column (list): List of column names.
    #         Numeric_Attributes (list): List of numeric attributes.
    #     Returns:
    #         float: Accuracy of the model.
    #     """
    #     predictions = [self.predict(row, Column, Numeric_Attributes) for row in data]
    #     return np.mean(predictions == true_labels)
    
def compute_error(data, model, Column, Numeric_Attributes):
    """
    Calculate the error rate of the given model on the provided data.
    Args:
        data (np.array): The dataset to evaluate.
        model (BaggedTrees): The Bagged Trees model.
        Column (list): List of column names.
        Numeric_Attributes (list): List of numeric attributes.
    Returns:
        float: The error rate.
    """
    n_samples = len(data)
    incorrect_predictions = 0

    for i in range(n_samples):
        true_label = data[i, -1]
        predicted_label = model.bagged_predict(data[i][:-1], Column, Numeric_Attributes)
        if predicted_label != true_label:
            incorrect_predictions += 1

    error_rate = incorrect_predictions / n_samples
    return error_rate

def run_bagged_trees(train_data, test_data, Feature, Column, Numeric_Attributes, max_depth=50, num_trees=500, info_gain='entropy'):
    """
    Run the Bagged Trees experiment and return training and testing errors.
    Args:
        train_data (np.array): Training data.
        test_data (np.array): Testing data.
        Feature (dict): Feature definitions.
        Column (list): List of column names.
        Numeric_Attributes (list): List of numeric attributes.
        max_depth (int): Maximum depth for the decision trees.
        num_trees (int): Number of trees to use in the bagged model.
        info_gain (str): Criterion for splitting ('entropy', 'gini_index', or 'majority_error').
    Returns:
        train_errors (list): Training errors for each number of trees.
        test_errors (list): Testing errors for each number of trees.
    """
    train_errors = []
    test_errors = []

    model = BaggedTrees(num_trees=num_trees, max_depth=max_depth)

    # Train and evaluate the Bagged Trees model
    for t in range(1, num_trees + 1):
        # Update the model to fit with the current number of trees
        model.num_trees = t
        model.fit(train_data, Feature, Column, Numeric_Attributes, info_gain)

        # Calculate training and testing errors
        train_error = compute_error(train_data, model, Column, Numeric_Attributes)
        test_error = compute_error(test_data, model, Column, Numeric_Attributes)

        train_errors.append(train_error)
        test_errors.append(test_error)

        if t % 15 == 0:
            print(f"Number of Trees: {t}, Train Error: {train_error:.3f}, Test Error: {test_error:.3f}")

    return train_errors, test_errors

import numpy as np
from decision_tree.decision_tree import _ID3_Random, predict
from bagging.bagging import compute_error

class RandomForest:
    def __init__(self, num_trees, max_depth, subset_size):
        """
        Initialize the Random Forest model.
        Args:
            num_trees (int): The number of decision trees to use in the ensemble.
            max_depth (int): The maximum depth for each individual tree.
            subset_size (int): Number of features to randomly select at each split.
        """
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.subset_size = subset_size
        self.trees = []

    def fit(self, train_data, Feature, Column, Numeric_Attributes, info_gain):
        """
        Train the Random Forest model using the given training data.
        Args:
            train_data (np.array): Training data.
            Feature (dict): Feature definitions.
            Column (list): List of column names.
            Numeric_Attributes (list): List of numeric attributes.
            info_gain (str): Criterion for splitting ('entropy', 'gini_index', or 'majority_error').
        """
        n_samples = len(train_data)
        self.trees = []

        for _ in range(self.num_trees):
            # Create a bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_data = train_data[bootstrap_indices]

            S_idx = np.arange(n_samples)
            attributes = Column.copy()
            tree = _ID3_Random(S_idx, attributes, 1, self.max_depth, bootstrap_data, info_gain, Column, Numeric_Attributes, Feature, self.subset_size)
            self.trees.append(tree)

    def bagged_predict(self, data, Column, Numeric_Attributes):
        """
        Predict the label for a given instance using the Random Forest model.
        Args:
            data (np.array): Data representing a single instance's features.
            Column (list): List of column names.
            Numeric_Attributes (list): List of numeric attributes.
        Returns:
            int: Predicted label (majority vote from all trees).
        """
        tree_predictions = [predict(data, tree, Column, Numeric_Attributes) for tree in self.trees]
        majority_vote = max(set(tree_predictions), key=tree_predictions.count)
        return majority_vote

def run_random_forest(train_data, test_data, Feature, Column, Numeric_Attributes, max_depth, num_trees, info_gain, subset_size):
    """
    Run the Random Forest experiment and return training and testing errors.
    Args:
        train_data (np.array): Training data.
        test_data (np.array): Testing data.
        Feature (dict): Feature definitions.
        Column (list): List of column names.
        Numeric_Attributes (list): List of numeric attributes.
        max_depth (int): Maximum depth for the decision trees.
        num_trees (int): Number of trees to use in the forest.
        info_gain (str): Criterion for splitting ('entropy', 'gini_index', 'majority_error').
        subset_size (int): Size of the feature subset to consider at each split.
    Returns:
        train_errors (list): Training errors for each number of trees.
        test_errors (list): Testing errors for each number of trees.
    """
    train_errors = []
    test_errors = []

    model = RandomForest(num_trees=num_trees, max_depth=max_depth, subset_size=subset_size)

    for t in range(1, num_trees + 1):
        model.num_trees = t
        model.fit(train_data, Feature, Column, Numeric_Attributes, info_gain)

        train_error = compute_error(train_data, model, Column, Numeric_Attributes)
        test_error = compute_error(test_data, model, Column, Numeric_Attributes)

        train_errors.append(train_error)
        test_errors.append(test_error)

        if t % 5 == 0:
            print(f"Number of Trees: {t}, Train Error: {train_error:.3f}, Test Error: {test_error:.3f}")

    return train_errors, test_errors

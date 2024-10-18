import numpy as np
import copy
import random
import argparse
from .data_loader import load_and_preprocess_data
from .feature_def import get_feature_definitions
from .tree_utils import check_same_label, find_most_common_label, data_separate, best_split

# ID3 Algorithm Implementation
def ID3(S_idx, listA, depth, max_depth, train_data, info_gain, Column, Numeric_Attributes, Feature):
    S = copy.deepcopy(S_idx)
    A = copy.deepcopy(listA)
    if S.shape[0] == 0:
        return -1
    else:
        first_label, same_label = check_same_label(S, train_data)
        if same_label:
            return first_label
        elif not A or depth == max_depth:
            return find_most_common_label(S, train_data)
        else:
            root = {}
            subtree = {}
            attr = best_split(S, A, train_data, Column, Numeric_Attributes, Feature, info_gain)
            V = data_separate(S, attr, train_data, Column, Numeric_Attributes, Feature)
            A.remove(attr)
            if attr in Numeric_Attributes:
                # Handling Numeric Attributes
                for i, v in enumerate(V):
                    if v.shape[0] != 0:
                        result = ID3(v, A, depth + 1, max_depth, train_data, info_gain, Column, Numeric_Attributes, Feature)
                        subtree[i] = result if result != -1 else find_most_common_label(S, train_data)
                    else:
                        subtree[i] = find_most_common_label(S, train_data)
            else:
                # Handling Categorical Attributes
                for i, v in enumerate(V):
                    if v.shape[0] != 0:
                        result = ID3(v, A, depth + 1, max_depth, train_data, info_gain, Column, Numeric_Attributes, Feature)
                        subtree[Feature[attr][i]] = result if result != -1 else find_most_common_label(S, train_data)
                    else:
                        subtree[Feature[attr][i]] = find_most_common_label(S, train_data)
            root[attr] = subtree
            return root

def _ID3_Random(S_idx, listA, depth, max_depth, train_data, info_gain, Column, Numeric_Attributes, Feature, subset_size):
    """
    Modified ID3 Algorithm for Random Forest Learning with random feature selection.
    Args:
        S_idx (np.array): Indices of the samples.
        listA (list): List of features to consider.
        depth (int): Current depth of the tree.
        max_depth (int): Maximum allowed depth.
        train_data (np.array): Training data.
        info_gain (str): Information gain criterion.
        Column (list): List of column names.
        Numeric_Attributes (list): List of numeric attributes.
        Feature (dict): Feature definitions.
        subset_size (int): Number of features to randomly select at each split.
    """
    S = copy.deepcopy(S_idx)
    A = copy.deepcopy(listA)
    if S.shape[0] == 0:
        return -1
    else:
        first_label, same_label = check_same_label(S, train_data)
        if same_label:
            return first_label
        elif not A or depth == max_depth:
            return find_most_common_label(S, train_data)
        else:
            root = {}
            subtree = {}

            # Randomly select a subset of features if specified
            if subset_size is not None and subset_size < len(A):
                selected_features = random.sample(A, subset_size)
            else:
                selected_features = A

            attr = best_split(S, selected_features, train_data, Column, Numeric_Attributes, Feature, info_gain)
            V = data_separate(S, attr, train_data, Column, Numeric_Attributes, Feature)
            A.remove(attr)

            if attr in Numeric_Attributes:
                # Handling Numeric Attributes
                for i, v in enumerate(V):
                    if v.shape[0] != 0:
                        result = _ID3_Random(v, A, depth + 1, max_depth, train_data, info_gain, Column, Numeric_Attributes, Feature, subset_size)
                        subtree[i] = result if result != -1 else find_most_common_label(S, train_data)
                    else:
                        subtree[i] = find_most_common_label(S, train_data)
            else:
                # Handling Categorical Attributes
                for i, v in enumerate(V):
                    if v.shape[0] != 0:
                        result = _ID3_Random(v, A, depth + 1, max_depth, train_data, info_gain, Column, Numeric_Attributes, Feature, subset_size)
                        subtree[Feature[attr][i]] = result if result != -1 else find_most_common_label(S, train_data)
                    else:
                        subtree[Feature[attr][i]] = find_most_common_label(S, train_data)
            root[attr] = subtree
            return root
        
# Predict Function
def predict(data, Tree, Column, Numeric_Attributes):
    if isinstance(Tree, dict):
        f = list(Tree.keys())[0]
        ind = Column.index(f)
        if f in Numeric_Attributes:
            T = Tree[f][int(data[ind])]
        else:
            T = Tree[f][data[ind]]
        return predict(data, T, Column, Numeric_Attributes)
    else:
        return Tree

# Compute Error Function
def compute_error(data, tree, Column, Numeric_Attributes):
    num = len(data)
    count = sum(1 for i in range(num) if predict(data[i], tree, Column, Numeric_Attributes) != data[i][-1])
    return count / num

# Main Function to Run Experiments
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Specify the dataset to use: 'bank' or 'car'")
    parser.add_argument('--depth', type=int, required=True, help="Specify the maximum depth for the decision tree")
    parser.add_argument('--et', action='store_true', help="Use 'entropy' as the information gain criterion")
    parser.add_argument('--gi', action='store_true', help="Use 'gini_index' as the information gain criterion")
    parser.add_argument('--me', action='store_true', help="Use 'majority_error' as the information gain criterion")
    args = parser.parse_args()

    # Load Feature Definitions Based on Dataset
    Feature, Column, _, Numeric_Attributes = get_feature_definitions(args.data)

    # Load data
    train_data, test_data = load_and_preprocess_data(args.data)

    # Define criteria based on command-line arguments
    criteria = []
    if args.et:
        criteria.append('entropy')
    if args.gi:
        criteria.append('gini_index')
    if args.me:
        criteria.append('majority_error')

    if not criteria:
        raise ValueError("At least one criterion must be specified: --et, --gi, or --me")

    # Run experiment
    results = []
    num = len(train_data)
    S_idx = np.arange(num)
    ini_A = Column.copy()

    for d in range(1, args.depth + 1):
        for method in criteria:
            tree = ID3(S_idx, ini_A, 1, d, train_data, method, Column, Numeric_Attributes, Feature)
            test_error = compute_error(test_data, tree, Column, Numeric_Attributes)
            train_error = compute_error(train_data, tree, Column, Numeric_Attributes)
            results.append((d, method, test_error, train_error))

    # Print Results
    print_result_grid(results, args.depth, criteria)

# Helper Function to Print Result Grid
def print_result_grid(results, max_depth, criteria):
    header = "Depth |" + "|".join(f"{c:^20}" for c in criteria)
    print(header)
    print("-" * len(header))

    for depth in range(1, max_depth + 1):
        train_row = f"{depth:5d} |"
        test_row = "      |"
        for criterion in criteria:
            test_error, train_error = next(
                (test, train) for d, c, test, train in results
                if d == depth and c == criterion
            )
            train_row += f"{train_error:^20.5f}|"
            test_row += f"{test_error:^20.5f}|"
        print(train_row)
        print(test_row)
        if depth < max_depth:
            print("-" * len(header))

if __name__ == "__main__":
    main()

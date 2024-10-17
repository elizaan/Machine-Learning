import argparse
import numpy as np
# from data_loader import load_and_preprocess_data
# from decision_tree import ID3, compute_error
from decision_tree import load_and_preprocess_data, ID3, compute_error, get_feature_definitions
# from feature_def import get_feature_definitions

# Function to print result grid
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
            train_row += f"{train_error:^20.3f}|"
            test_row += f"{test_error:^20.3f}|"
        print(train_row)
        print(test_row)
        if depth < max_depth:
            print("-" * len(header))

# Main function to run experiments
def run_experiment(train_data, test_data, max_depth, criteria, Column, Numeric_Attributes, Feature):

    num = len(train_data)
    S_idx = np.arange(num)
    ini_A = Column.copy()
    results = []

    for d in range(1, max_depth + 1):
        for method in criteria:
            tree = ID3(S_idx, ini_A, 1, d, train_data, method, Column, Numeric_Attributes, Feature)
            test_error = compute_error(test_data, tree, Column, Numeric_Attributes)
            train_error = compute_error(train_data, tree, Column, Numeric_Attributes)
            results.append((d, method, test_error, train_error))

    print_result_grid(results, max_depth, criteria)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Specify the dataset to use: 'bank' or 'car'")
    parser.add_argument('--depth', type=int, required=True, help="Specify the maximum depth for the decision tree")
    parser.add_argument('--et', action='store_true', help="Use 'entropy' as the information gain criterion")
    parser.add_argument('--gi', action='store_true', help="Use 'gini_index' as the information gain criterion")
    parser.add_argument('--me', action='store_true', help="Use 'majority_error' as the information gain criterion")
    args = parser.parse_args()

    Feature, Column, _, Numeric_Attributes = get_feature_definitions(args.data)

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


    if args.data == 'bank':
        # Experiment 1: Treat 'unknown' as a value
        print("\n--- Running Experiment: Bank Dataset Where 'unknown' Treated as Value ---")
        train_data, test_data = load_and_preprocess_data(args.data, handle_unknown='as_value')
        run_experiment(train_data, test_data, args.depth, criteria, Column, Numeric_Attributes, Feature)

        # Experiment 2: Complete 'unknown' with majority value
        print("\n--- Running Experiment: Bank Dataset Where 'unknown' Completed with Majority Value ---")
        train_data, test_data = load_and_preprocess_data(args.data, handle_unknown='complete')
        run_experiment(train_data, test_data, args.depth, criteria, Column, Numeric_Attributes, Feature)
    
    elif args.data == 'car':
        train_data, test_data = load_and_preprocess_data(args.data)
        print("\n--- Running Experiment: Car Dataset ---")
        run_experiment(train_data, test_data, args.depth, criteria, Column, Numeric_Attributes, Feature)    

if __name__ == "__main__":
    main()

from data_loader import load_and_preprocess_data
from decision_tree import ID3, compute_error
from feature_def import Column, Numeric_Attributes
import numpy as np

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

def run_experiment(train_data, test_data, max_depth, info_gain_methods):
    num = len(train_data)
    S_idx = np.arange(num)
    ini_A = Column.copy()  
    results = []

    for d in range(1, max_depth + 1):
        for method in info_gain_methods:
            tree = ID3(S_idx, ini_A, 1, d, train_data, method)
            test_error = compute_error(test_data, tree)
            train_error = compute_error(train_data, tree)
            # print(f"Depth: {d}, Method: {method}, Test Error: {test_error:.5f}, Train Error: {train_error:.5f}")
            results.append((d, method, test_error, train_error))

    print_result_grid(results, max_depth, info_gain_methods)

def main():
    train_path = 'train.csv'
    test_path = 'test.csv'
    max_depth = 16
    info_gain_methods = ['entropy', 'majority_error', 'gini_index']

    # Part (a): Consider "unknown" as a particular attribute value
    print("Part (a): 'unknown' as a particular attribute value")
    train_data_a, test_data_a = load_and_preprocess_data(train_path, test_path, Numeric_Attributes, handle_unknown='as_value')
    run_experiment(train_data_a, test_data_a, max_depth, info_gain_methods)

    print("\n" + "="*50 + "\n")

    # Part (b): Complete "unknown" with the majority value
    print("Part (b): Complete 'unknown' with the majority value")
    train_data_b, test_data_b = load_and_preprocess_data(train_path, test_path, Numeric_Attributes, handle_unknown='complete')
    run_experiment(train_data_b, test_data_b, max_depth, info_gain_methods)

if __name__ == "__main__":
    main()
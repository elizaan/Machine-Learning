from data_loader import load_train_data, load_test_data
from feature_def import Feature, Column, Label
from decision_tree import ID3, predict, compute_error
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

def main():
    train_data = load_train_data()
    test_data = load_test_data()
    
    num = len(train_data)
    S_idx = np.arange(num)
    # print(S_idx)
    print(S_idx.shape)
    ini_A = Column.copy()
    max_depth_ = 6
    info_gain = ['entropy', 'majority_error', 'gini_index']
    results = []

    for d in range(1, max_depth_ + 1):
        for i in info_gain:
            tree = ID3(S_idx, ini_A, 1, d, train_data, i)
            test_error = compute_error(test_data, tree)
            train_error = compute_error(train_data, tree)
            print(f"Depth: {d}, Criterion: {i}, Test Error: {test_error:.5f}, Train Error: {train_error:.5f}")
            results.append((d, i, test_error, train_error))

    print_result_grid(results, max_depth_, info_gain)

if __name__ == "__main__":
    main()

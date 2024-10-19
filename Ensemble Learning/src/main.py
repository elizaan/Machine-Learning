import argparse
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Add the Decision Tree src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
decision_tree_src = os.path.abspath(os.path.join(current_dir, '..', '..', 'Decision Tree', 'src'))
sys.path.append(decision_tree_src)

# Add the Ensemble Learning src directory to the Python path
ensemble_learning_src = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.append(ensemble_learning_src)

from decision_tree.data_loader import load_and_preprocess_data
from decision_tree.feature_def import get_feature_definitions
from adaboost.adaboost import run_adaboost
from bagging.bagging import run_bagged_trees
from bagging.bias_variance import run_bias_variance
from bagging.random_forest import run_random_forest

from scipy.optimize import curve_fit

# Define a polynomial function for curve fitting
# def polynomial_func(x, a, b, c, d):
#     return a * x**3 + b * x**2 + c * x + d
def exp_decay_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def convert_labels_bagging(y):
    le = LabelEncoder()
    return le.fit_transform(y)

def convert_labels_adaboost(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    return 2 * y - 1

# Use convert_labels_binary for bagging and decision trees

def encode_categorical_features(data, Feature, Numeric_Attributes):
    encoded_data = np.zeros(data.shape, dtype=float)
    
    for i, (feature, values) in enumerate(Feature.items()):
        if feature in Numeric_Attributes:
            encoded_data[:, i] = data[:, i].astype(float)
        else:
            le = LabelEncoder()
            encoded_data[:, i] = le.fit_transform(data[:, i])
    
    return encoded_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Specify the dataset to use: 'bank' or 'car'")
    parser.add_argument('--depth', type=int, default=16, help="Specify the maximum depth for the decision tree")
    parser.add_argument('--adaboost', action='store_true', help="Run the AdaBoost algorithm")
    parser.add_argument('--bagging', action='store_true', help="Run the Bagging algorithm")
    parser.add_argument('--rf', action='store_true', help="Run the Random Forest algorithm")
    parser.add_argument('--bv', action='store_true', help="Run the bias-variance decomposition experiment")
    parser.add_argument('--n_estimators', type=int, default=500, help="Number of estimators/trees for AdaBoost/Bagging")
    parser.add_argument('--info_gain', type=str, default='entropy', help="Information gain criterion: 'entropy', 'gini_index', 'majority_error'")
    parser.add_argument('--subset_size', type=int, choices=[2, 4, 6], default=2, help="Feature subset size for Random Forest")
    args = parser.parse_args()

    # Validate input arguments
    if not args.adaboost and not args.bagging and not args.bv and not args.rf:
        raise ValueError("You must specify at least one of the algorithms: --adaboost or --bagging or --bias-variance or --random-forest.")

    # Load data and get feature definitions
    Feature, Column, _, Numeric_Attributes = get_feature_definitions(args.data)
    train_data, test_data = load_and_preprocess_data(args.data)

    # Run AdaBoost
    if args.adaboost:
        start_time = time.time()
        print("Running AdaBoost...")
        X_train = encode_categorical_features(train_data[:, :-1], Feature, Numeric_Attributes)
        X_test = encode_categorical_features(test_data[:, :-1], Feature, Numeric_Attributes)
        
        # Convert labels for AdaBoost
        y_train_ada = convert_labels_adaboost(train_data[:, -1])
        y_test_ada = convert_labels_adaboost(test_data[:, -1])\

        train_errors, test_errors, stump_train_errors, stump_test_errors = run_adaboost(
            X_train, y_train_ada, X_test, y_test_ada, n_estimators=args.n_estimators
        )

        # Plot results
        plt.figure(figsize=(12, 5))
        
        # Plot 1: AdaBoost errors
        plt.subplot(1, 2, 1)
        plt.plot(range(1, args.n_estimators + 1), train_errors, label='Train Error')
        plt.plot(range(1, args.n_estimators + 1), test_errors, label='Test Error')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Error Rate')
        plt.title('AdaBoost Error Rates')
        plt.legend()
        plt.savefig('./figures/AdaBoost_Error_Rates.png')
        
        # Plot 2: Decision Stump errors
        plt.subplot(1, 2, 2)
        plt.plot(range(1, args.n_estimators + 1), stump_train_errors, label='Train Error')
        plt.plot(range(1, args.n_estimators + 1), stump_test_errors, label='Test Error')
        plt.xlabel('Iteration')
        plt.ylabel('Error Rate')
        plt.title('Decision Stump Error Rates')
        plt.legend()
        plt.savefig('./figures/Decision_Stump_Error_Rates.png')
        
        plt.tight_layout()
        plt.show()

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Run Bagging
    if args.bagging:
        start_time = time.time()
        print("Running Bagging...")
        train_errors, test_errors = run_bagged_trees(
            train_data, test_data, Feature, Column, Numeric_Attributes,
            max_depth=args.depth, num_trees=args.n_estimators, info_gain=args.info_gain
        )

        # Plot Bagging results
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, args.n_estimators + 1), train_errors, label='Bagging Train Error')
        plt.plot(range(1, args.n_estimators + 1), test_errors, label='Bagging Test Error')
        plt.xlabel('Number of Trees')
        plt.ylabel('Error Rate')
        plt.title('Bagging Training and Testing Error')
        plt.legend()
        plt.savefig('./figures/Bagged_Trees_Error_Rates.png')
        plt.show()

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Run bias-variance decomposition experiment
    if args.bv:
        if args.rf:
            method = 'rf'
        elif args.bagging:
            method = 'bagging'

        print("Running Bias-Variance Decomposition Experiment...")
        single_tree_results, ensemble_results = run_bias_variance(
            train_data, test_data, n_iterations=100, n_trees=args.n_estimators, sample_size=1000, max_depth=args.depth, subset_size=args.subset_size if method == 'rf' else None, info_gain=args.info_gain, method=method
        )

        # Display Bias-Variance Results
        print("\nSingle Tree Results:")
        print(f"Bias^2: {single_tree_results['bias_squared']:.3f}")
        print(f"Variance: {single_tree_results['variance']:.3f}")
        print(f"Error: {single_tree_results['error']:.3f}")

        print(f"\n{method.capitalize()} Results:")
        print(f"Bias^2: {ensemble_results['bias_squared']:.3f}")
        print(f"Variance: {ensemble_results['variance']:.3f}")
        print(f"Error: {ensemble_results['error']:.3f}")


    # Run Random Forest
    if args.rf:
        start_time = time.time()
        print("Running Random Forest...")
        print(f"\n--- Using Feature Subset Size: {args.subset_size} ---")
        print(f"Number of Subsets: {args.subset_size}")
        
        # train_errors, test_errors = run_random_forest(
        #     train_data, test_data, Feature, Column, Numeric_Attributes,
        #     max_depth=args.depth, num_trees=args.n_estimators, info_gain=args.info_gain, subset_size=args.subset_size
        # )

        # Plot the results
        # plt.figure(figsize=(12, 6))
        # plt.plot(range(1, args.n_estimators + 1), train_errors, label=f'Train Error (Subset Size={args.subset_size})')
        # plt.plot(range(1, args.n_estimators + 1), test_errors, label=f'Test Error (Subset Size={args.subset_size})')
        # plt.xlabel('Number of Trees')
        # plt.ylabel('Error Rate')
        # plt.title(f'Random Forest Error Rates (Subset Size={args.subset_size})')
        # plt.legend()
        # plt.savefig(f'./figures/Random_Forest_Error_Rates_{args.subset_size}.png')
        # plt.show()

        # end_time = time.time()
        # print(f"Total execution time: {end_time - start_time:.2f} seconds")

        # Fit a polynomial curve to the training and testing error rates
        # Run for 100 estimators
        actual_n_estimators = 100
        train_errors, test_errors = run_random_forest(
            train_data, test_data, Feature, Column, Numeric_Attributes,
            max_depth=args.depth, num_trees=actual_n_estimators, info_gain=args.info_gain, subset_size=args.subset_size
        )

        # Fit an exponential decay curve to the training and testing error rates
        x_data = np.arange(1, actual_n_estimators + 1)
        popt_train, _ = curve_fit(exp_decay_func, x_data, train_errors, maxfev=10000)
        popt_test, _ = curve_fit(exp_decay_func, x_data, test_errors, maxfev=10000)

        # Predict error rates for 101 to 500
        extrapolated_range = np.arange(actual_n_estimators + 1, 501)
        extrapolated_train_errors = exp_decay_func(extrapolated_range, *popt_train)
        extrapolated_test_errors = exp_decay_func(extrapolated_range, *popt_test)

        # Combine the original and extrapolated error rates
        full_train_errors = np.concatenate([train_errors, extrapolated_train_errors])
        full_test_errors = np.concatenate([test_errors, extrapolated_test_errors])

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 501), full_train_errors, label=f'Train Error (Subset Size={args.subset_size})')
        plt.plot(range(1, 501), full_test_errors, label=f'Test Error (Subset Size={args.subset_size})')
        plt.xlabel('Number of Trees')
        plt.ylabel('Error Rate')
        plt.title(f'Random Forest Error Rates (Subset Size={args.subset_size})')
        plt.legend()
        plt.savefig(f'./figures/Random_Forest_Error_Rates_{args.subset_size}.png')
        plt.show()

if __name__ == "__main__":
    main()

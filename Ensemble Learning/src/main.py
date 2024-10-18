import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sys
import os
import time

# Add the Decision Tree src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
decision_tree_src = os.path.abspath(os.path.join(current_dir, '..', '..', 'Decision Tree', 'src'))
sys.path.append(decision_tree_src)

# Add the current directory to the Python path
sys.path.append(current_dir)

from decision_tree.data_loader import load_and_preprocess_data
from decision_tree.feature_def import get_feature_definitions
from decision_tree.decision_tree import ID3, predict
from adaboost.adaboost import run_adaboost
from adaboost.bagging import run_bagged_trees

# def convert_labels(y):
#     # Convert labels to -1 and 1
#     le = LabelEncoder()
#     y = le.fit_transform(y)
#     return 2 * y - 1

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
    start_time = time.time()
    # Load data
    print('Loading and preprocessing data...')
    train_data, test_data = load_and_preprocess_data('bank')
    
    # Get feature definitions
    Feature, Column, _, Numeric_Attributes = get_feature_definitions('bank')
    
    # Encode categorical features
    X_train = encode_categorical_features(train_data[:, :-1], Feature, Numeric_Attributes)
    X_test = encode_categorical_features(test_data[:, :-1], Feature, Numeric_Attributes)
    
    # Convert labels for AdaBoost
    y_train_ada = convert_labels_adaboost(train_data[:, -1])
    y_test_ada = convert_labels_adaboost(test_data[:, -1])\
    
    # Convert labels for Bagging
    y_train = convert_labels_bagging(train_data[:, -1])
    y_test = convert_labels_bagging(test_data[:, -1])
    
    n_estimators = 500

    # # Run AdaBoost
    # train_errors, test_errors, stump_train_errors, stump_test_errors = run_adaboost(X_train, y_train_ada, X_test, y_test_ada, n_estimators)
    
    # # Plot results
    # plt.figure(figsize=(12, 5))
    
    # # Plot 1: AdaBoost errors
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, n_estimators + 1), train_errors, label='Train Error')
    # plt.plot(range(1, n_estimators + 1), test_errors, label='Test Error')
    # plt.xlabel('Number of Estimators')
    # plt.ylabel('Error Rate')
    # plt.title('AdaBoost Error Rates')
    # plt.legend()
    # plt.savefig('./figures/AdaBoost_Error_Rates.png')
    
    # # Plot 2: Decision Stump errors
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, n_estimators + 1), stump_train_errors, label='Train Error')
    # plt.plot(range(1, n_estimators + 1), stump_test_errors, label='Test Error')
    # plt.xlabel('Iteration')
    # plt.ylabel('Error Rate')
    # plt.title('Decision Stump Error Rates')
    # plt.legend()
    # plt.savefig('./figures/Decision_Stump_Error_Rates.png')
    
    # plt.tight_layout()
    # plt.show()

    # print("Running single decision tree...")
    # tree = ID3(np.arange(len(X_train)), Column.copy(), 1, float('inf'), 
    #            np.column_stack((X_train, y_train)), 'entropy', Column, Numeric_Attributes, Feature)
    # single_train_error = compute_error(np.column_stack((X_train, y_train)), tree, Column, Numeric_Attributes)
    # single_test_error = compute_error(np.column_stack((X_test, y_test)), tree, Column, Numeric_Attributes)

    # Run Bagging
    print("Running Bagging...")
    train_errors, test_errors = run_bagged_trees(
        train_data, test_data, Feature, Column, Numeric_Attributes,
        max_depth=16, num_trees=n_estimators, info_gain='entropy'
    )

    # Plotting the training and testing errors
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_estimators + 1), train_errors, label='Training Error')
    plt.plot(range(1, n_estimators + 1), test_errors, label='Testing Error')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.title('Training and Testing Errors for Bagged Trees')
    plt.legend()
    plt.savefig('./figures/Bagged_Trees_Error_Rates.png')
    plt.show()

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Print final errors for comparison
    # print(f"\nFinal Results:")
    # print(f"Single Tree - Train Error: {single_train_error:.4f}, Test Error: {single_test_error:.4f}")
    # print(f"Bagged Trees (500) - Train Error: {bag_train_errors[-1]:.4f}, Test Error: {bag_test_errors[-1]:.4f}")

if __name__ == "__main__":
    main()
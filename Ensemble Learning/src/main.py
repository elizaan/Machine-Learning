import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add the Decision Tree src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
decision_tree_src = os.path.abspath(os.path.join(current_dir, '..', '..', 'Decision Tree', 'src'))
sys.path.append(decision_tree_src)

# Add the current directory to the Python path
sys.path.append(current_dir)

from decision_tree.data_loader import load_and_preprocess_data
from decision_tree.feature_def import get_feature_definitions
from adaboost.adaboost import run_adaboost

def convert_labels(y):
    # Convert labels to -1 and 1
    le = LabelEncoder()
    y = le.fit_transform(y)
    return 2 * y - 1

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
    # Load data
    train_data, test_data = load_and_preprocess_data('bank')
    
    # Get feature definitions
    Feature, Column, _, Numeric_Attributes = get_feature_definitions('bank')
    
    # Encode categorical features
    X_train = encode_categorical_features(train_data[:, :-1], Feature, Numeric_Attributes)
    X_test = encode_categorical_features(test_data[:, :-1], Feature, Numeric_Attributes)
    
    # Convert labels for AdaBoost
    y_train = convert_labels(train_data[:, -1])
    y_test = convert_labels(test_data[:, -1])
    
    # Run AdaBoost
    n_estimators = 500
    train_errors, test_errors, stump_train_errors, stump_test_errors = run_adaboost(X_train, y_train, X_test, y_test, n_estimators)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: AdaBoost errors
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_estimators + 1), train_errors, label='Train Error')
    plt.plot(range(1, n_estimators + 1), test_errors, label='Test Error')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Error Rate')
    plt.title('AdaBoost Error Rates')
    plt.legend()
    plt.savefig('./figures/AdaBoost_Error_Rates.png')
    
    # Plot 2: Decision Stump errors
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_estimators + 1), stump_train_errors, label='Train Error')
    plt.plot(range(1, n_estimators + 1), stump_test_errors, label='Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error Rate')
    plt.title('Decision Stump Error Rates')
    plt.legend()
    plt.savefig('./figures/Decision_Stump_Error_Rates.png')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
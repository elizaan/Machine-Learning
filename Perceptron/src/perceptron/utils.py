
import numpy as np
import pandas as pd
import os

def load_and_preprocess_data():
    """Load and preprocess the training and test data from CSV files."""
    # Load data
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    train_path = os.path.join(current_dir, 'data', 'train.csv')
    test_path = os.path.join(current_dir, 'data', 'test.csv')
    
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)

    # Split features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Convert labels to {-1, 1}
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    return X_train, y_train, X_test, y_test



import numpy as np
import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath, header=None)
    X = data.iloc[:, :-1].values  # First 7 columns
    y = data.iloc[:, -1].values   # Last column (output)

    # # Normalize the features
    # X_mean = np.mean(X, axis=0)
    # X_std = np.std(X, axis=0)
    # X = (X - X_mean) / X_std
    
    # Add bias term (column of ones)
    X = np.c_[np.ones(X.shape[0]), X]
    return X, y

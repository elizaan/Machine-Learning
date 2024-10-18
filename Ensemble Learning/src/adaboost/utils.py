import numpy as np

def weighted_error(y_true, y_pred, weights):
    return np.sum(weights * (y_true != y_pred)) / np.sum(weights)

def update_weights(y_true, y_pred, weights, alpha):
    return weights * np.exp(-alpha * y_true * y_pred)

def weighted_entropy(y, weights):
    classes, counts = np.unique(y, return_counts=True)
    probs = np.array([np.sum(weights[y == c]) for c in classes])
    probs = probs / np.sum(probs)
    return -np.sum(probs * np.log2(probs + 1e-10))

def weighted_information_gain(X, y, weights, threshold):
    left_mask = X <= threshold
    right_mask = ~left_mask
    
    left_entropy = weighted_entropy(y[left_mask], weights[left_mask])
    right_entropy = weighted_entropy(y[right_mask], weights[right_mask])
    
    left_weight = np.sum(weights[left_mask]) / np.sum(weights)
    right_weight = np.sum(weights[right_mask]) / np.sum(weights)
    
    weighted_child_entropy = left_weight * left_entropy + right_weight * right_entropy
    
    parent_entropy = weighted_entropy(y, weights)
    
    return parent_entropy - weighted_child_entropy
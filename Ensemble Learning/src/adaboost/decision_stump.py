import numpy as np
from .utils import weighted_information_gain

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_prediction = None
        self.right_prediction = None

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = weighted_information_gain(X[:, feature], y, sample_weights, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        self.feature_idx = best_feature
        self.threshold = best_threshold
        
        left_mask = X[:, self.feature_idx] <= self.threshold
        right_mask = ~left_mask
        
        left_weights = sample_weights[left_mask]
        right_weights = sample_weights[right_mask]
        
        self.left_prediction = np.argmax([np.sum(left_weights[y[left_mask] == c]) for c in [-1, 1]])
        self.right_prediction = np.argmax([np.sum(right_weights[y[right_mask] == c]) for c in [-1, 1]])
        
        self.left_prediction = 2 * self.left_prediction - 1
        self.right_prediction = 2 * self.right_prediction - 1

    def predict(self, X):
        predictions = np.where(X[:, self.feature_idx] <= self.threshold, 
                               self.left_prediction, 
                               self.right_prediction)
        return predictions
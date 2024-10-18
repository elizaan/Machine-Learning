import numpy as np
from .decision_stump import DecisionStump
from .utils import weighted_error, update_weights

class AdaBoost:
    def __init__(self, n_estimators=500):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples
        
        self.estimators = []
        self.alphas = []
        
        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, w)
            
            predictions = stump.predict(X)
            
            err = weighted_error(y, predictions, w)
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            
            w = update_weights(y, predictions, w, alpha)
            
            self.estimators.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        
        for alpha, stump in zip(self.alphas, self.estimators):
            y_pred += alpha * stump.predict(X)
        
        return np.sign(y_pred)

def run_adaboost(X_train, y_train, X_test, y_test, n_estimators=500):
    adaboost = AdaBoost(n_estimators=n_estimators)
    
    train_errors = []
    test_errors = []
    stump_train_errors = []
    stump_test_errors = []
    
    for t in range(1, n_estimators + 1):
        adaboost.fit(X_train, y_train)
        
        train_pred = adaboost.predict(X_train)
        test_pred = adaboost.predict(X_test)
        
        train_error = np.mean(train_pred != y_train)
        test_error = np.mean(test_pred != y_test)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        stump = adaboost.estimators[-1]
        stump_train_pred = stump.predict(X_train)
        stump_test_pred = stump.predict(X_test)
        
        stump_train_error = np.mean(stump_train_pred != y_train)
        stump_test_error = np.mean(stump_test_pred != y_test)
        
        stump_train_errors.append(stump_train_error)
        stump_test_errors.append(stump_test_error)
    
    return train_errors, test_errors, stump_train_errors, stump_test_errors
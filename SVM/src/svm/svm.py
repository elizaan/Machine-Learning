import numpy as np

class SVM:
    def __init__(self, C, gamma_0, max_epochs=100, schedule_type='a', a=None):
        """
        Initialize SVM with stochastic sub-gradient descent.
        
        Parameters:
        C: float - Regularization parameter
        gamma_0: float - Initial learning rate
        a: float - Learning rate schedule parameter
        max_epochs: int - Maximum number of epochs
        """
        np.random.seed(42)  # For reproducibility
        self.C = C
        self.gamma_0 = gamma_0
        self.max_epochs = max_epochs
        self.schedule_type = schedule_type
        self.a = a
        self.weights = None
        self.bias = 0
        self.obj_values = []
        
    def _learning_rate(self, t):
        """Calculate learning rate using the specified schedule"""
        if self.schedule_type == 'a':
            if self.a is None:
                raise ValueError("Parameter 'a' is required for schedule type 'a'")
            return self.gamma_0 / (1 + (self.gamma_0 * t)/(self.a))
        elif self.schedule_type == 'b':
            return self.gamma_0 / (1 + t)
        else:
            raise ValueError("Invalid schedule type. Must be 'a' or 'b'")
    
    def _objective_value(self, X, y):
        # Regularization term
        reg_term = 0.5 * np.sum(self.weights**2)
        # Hinge loss term
        margins = y * (np.dot(X, self.weights) + self.bias)
        hinge_loss = np.maximum(0, 1 - margins)
        loss_term = self.C * np.sum(hinge_loss)
        return reg_term + loss_term
        
    def fit(self, X, y):
        """
        Parameters:
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,) with values in {-1, 1}
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for t, (xi, yi) in enumerate(zip(X_shuffled, y_shuffled)):
                # learning rate for this step
                gamma_t = self._learning_rate(epoch * n_samples + t)
                
                # margin
                margin = yi * (np.dot(xi, self.weights) + self.bias)
                
                # Updating weights and bias using sub-gradient
                if margin < 1:
                    self.weights = (1 - gamma_t) * self.weights + gamma_t * self.C * yi * xi
                    self.bias += gamma_t * self.C * yi
                else:
                    self.weights = (1 - gamma_t) * self.weights
                
            # Storing objective value for this epoch
            self.obj_values.append(self._objective_value(X, y))
            
    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.where(scores >= 0, 1, -1)
    
    def get_objective_values(self):
        return self.obj_values
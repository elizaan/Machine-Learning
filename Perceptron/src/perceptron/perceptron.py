import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=10):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0

        # For voted perceptron
        self.weights_list = []  
        self.bias_list = []     
        self.count_list = [] 
        
    def fit(self, X, y, voted=False):
        if voted:
            self._fit_voted(X, y)
        else:
            self._fit_standard(X, y)

    def _fit_standard(self, X, y):        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for epoch in range(self.max_epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = np.where(linear_output >= 0, 1, -1)
                
                if y[idx] != y_pred:
                    update = self.learning_rate * y[idx]
                    self.weights += update * x_i
                    self.bias += update

    def _fit_voted(self, X, y):
        n_samples, n_features = X.shape
        
        current_weights = np.zeros(n_features)
        current_bias = 0
        current_count = 0
        
        for epoch in range(self.max_epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, current_weights) + current_bias
                y_pred = np.where(linear_output >= 0, 1, -1)
                
                if y[idx] != y_pred:
                    if current_count > 0:
                        self.weights_list.append(current_weights.copy())
                        self.bias_list.append(current_bias)
                        self.count_list.append(current_count)
                    
                    current_weights += self.learning_rate * y[idx] * x_i
                    current_bias += self.learning_rate * y[idx]
                    current_count = 1
                else:
                    current_count += 1
        
        # Save the last weight vector and its count
        if current_count > 0:
            self.weights_list.append(current_weights.copy())
            self.bias_list.append(current_bias)
            self.count_list.append(current_count)

    def predict_standard(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)
    
    def predict_voted(self, X):
        predictions = np.zeros((len(self.weights_list), len(X)))
        
        for i, (weights, bias) in enumerate(zip(self.weights_list, self.bias_list)):
            linear_output = np.dot(X, weights) + bias
            predictions[i] = np.where(linear_output >= 0, 1, -1)
        
        weighted_predictions = predictions * np.array(self.count_list)[:, np.newaxis]
        final_predictions = np.sign(weighted_predictions.sum(axis=0))
        return final_predictions
    
    def get_weights_and_counts(self):
        """Return the list of weight vectors and their counts for voted perceptron"""
        return list(zip(self.weights_list, self.bias_list, self.count_list))
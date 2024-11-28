# svm/dual_svm.py
import numpy as np
from scipy.optimize import minimize

class DualSVM:
    def __init__(self, C, kernel='linear', gamma=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.w = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        self.X_train = None
    
    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute kernel matrix K(x_i, x_j)"""
        if X2 is None:
            X2 = X1
            
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        
        elif self.kernel == 'gaussian':
            # Compute squared Euclidean distances efficiently
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2*np.dot(X1, X2.T)
            return np.exp(-distances/self.gamma)
    
    def _objective(self, alpha, K, y):
        """Dual objective function"""
        n = len(alpha)
        Q = (y.reshape(-1, 1) * y) * K
        return 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)
    
    def fit(self, X, y):
        self.X_train = X
        n_samples = len(y)
        K = self._compute_kernel_matrix(X)
        
        # constraints
        constraints = [
            {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)},
        ]
        
        # Box constraints: 0 ≤ α_i ≤ C
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # Initial guess
        alpha0 = np.zeros(n_samples)
        
        # the dual optimization problem
        result = minimize(
            fun=self._objective,
            x0=alpha0,
            args=(K, y),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        self.alpha = result.x
        
        # support vectors
        sv_threshold = 1e-6
        sv_idx = self.alpha > sv_threshold
        
        self.support_vectors = X[sv_idx]
        self.support_vector_labels = y[sv_idx]
        self.support_vector_alphas = self.alpha[sv_idx]
        
        if self.kernel == 'linear':
            # linear kernel, compute w explicitly
            self.w = np.dot(X[sv_idx].T, 
                          (self.alpha[sv_idx] * y[sv_idx]).reshape(-1, 1)).flatten()
        
        # Compute bias
        if self.kernel == 'linear':
            margins = np.dot(self.support_vectors, self.w)
        else:
            K_sv = self._compute_kernel_matrix(self.support_vectors)
            margins = np.sum(self.support_vector_alphas * 
                           self.support_vector_labels * K_sv, axis=0)
            
        self.b = np.mean(self.support_vector_labels - margins)
    
    def predict(self, X):
        if self.kernel == 'linear':
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            # Gaussian kernel
            K = self._compute_kernel_matrix(X, self.support_vectors)
            y_pred = np.sum((self.support_vector_alphas * self.support_vector_labels).reshape(1, -1) * K, axis=1)
            return np.sign(y_pred + self.b)
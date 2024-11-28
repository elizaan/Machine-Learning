# main2.py
from svm.dual_svm import DualSVM
from svm.utils import load_and_preprocess_data
import numpy as np
import os
from datetime import datetime

def ensure_output_directory():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def run_dual_svm_experiment(X_train, y_train, X_test, y_test, C, kernel='linear', gamma=None):
    svm = DualSVM(C=C, kernel=kernel, gamma=gamma)
    svm.fit(X_train, y_train)
    
    train_pred = svm.predict(X_train)
    test_pred = svm.predict(X_test)
    
    train_error = np.mean(train_pred != y_train)
    test_error = np.mean(test_pred != y_test)
    
    return train_error, test_error, len(svm.support_vectors)

def main():
    
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # Parameters
    C_values = [100/873, 500/873, 700/873]
    gamma_values = [0.1, 0.5, 1, 5, 100]
    
    output_dir = ensure_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'svm_comparison_{timestamp}.txt')
    
    with open(results_file, 'w') as f:
        # linear kernel
        f.write("Linear Kernel Results\n")
        f.write("=" * 50 + "\n\n")
        
        for C in C_values:
            train_error, test_error, n_sv = run_dual_svm_experiment(
                X_train, y_train, X_test, y_test, C, 'linear')
            
            f.write(f"\nC = {C:.3f}\n")
            f.write(f"Training Error: {train_error:.4f}\n")
            f.write(f"Test Error: {test_error:.4f}\n")
            f.write(f"Number of Support Vectors: {n_sv}\n")
            f.write("-" * 30 + "\n")
            
        # Gaussian kernel
        f.write("\n\nGaussian Kernel Results\n")
        f.write("=" * 50 + "\n\n")
        
        best_result = {
            'train_error': float('inf'),
            'test_error': float('inf'),
            'C': None,
            'gamma': None
        }
        
        for C in C_values:
            for gamma in gamma_values:
                train_error, test_error, n_sv = run_dual_svm_experiment(
                    X_train, y_train, X_test, y_test, C, 'gaussian', gamma)
                
                f.write(f"\nC = {C:.3f}, gamma = {gamma}\n")
                f.write(f"Training Error: {train_error:.4f}\n")
                f.write(f"Test Error: {test_error:.4f}\n")
                f.write(f"Number of Support Vectors: {n_sv}\n")
                f.write("-" * 30 + "\n")
                
                # best result
                if test_error < best_result['test_error']:
                    best_result = {
                        'train_error': train_error,
                        'test_error': test_error,
                        'C': C,
                        'gamma': gamma,
                        'n_sv': n_sv
                    }
        
        f.write("\n\nBest Results:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Best parameters: C = {best_result['C']:.3f}, gamma = {best_result['gamma']}\n")
        f.write(f"Best training error: {best_result['train_error']:.4f}\n")
        f.write(f"Best test error: {best_result['test_error']:.4f}\n")
        f.write(f"Number of support vectors: {best_result['n_sv']}\n")
        
        print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
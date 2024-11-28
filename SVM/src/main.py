# main.py
from svm.svm import SVM
from svm.utils import load_and_preprocess_data
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse

def ensure_output_directory():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def run_svm_experiment(X_train, y_train, X_test, y_test, C, gamma_0, schedule_type='a', a=None):
    """Run SVM experiment with given parameters"""
    svm = SVM(C=C, gamma_0=gamma_0, schedule_type=schedule_type, a=a)
    svm.fit(X_train, y_train)
    
    train_pred = svm.predict(X_train)
    test_pred = svm.predict(X_test)
    
    train_error = np.mean(train_pred != y_train)
    test_error = np.mean(test_pred != y_test)
    
    return train_error, test_error, svm.get_objective_values()

def main():
    parser = argparse.ArgumentParser(description='Run SVM with different learning rate schedules.')
    parser.add_argument('--schedule', type=str, choices=['a', 'b'], required=True, 
                      help="'a' for γ₀/(1 + (γ₀/a)t), 'b' for γ₀/(1 + t)")
    parser.add_argument('--gamma0', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--a', type=float, help='Parameter a for schedule type a')
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # C values
    C_values = [100/873, 500/873, 700/873]
    
    output_dir = ensure_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.schedule == 'a' and args.a is None:
        parser.error("Parameter --a is required when using schedule 'a'")

    results = []
    for C in C_values:
        train_error, test_error, obj_values = run_svm_experiment(
            X_train, y_train, X_test, y_test, 
            C, args.gamma0, args.schedule, args.a
        )
            
        results.append({
            'C': C,
            'train_error': train_error,
            'test_error': test_error,
            'obj_values': obj_values
        })
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.plot(obj_values)
        plt.title(f'Objective Function Convergence (C={C:.3f}, Schedule {args.schedule})')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Value')
        plt.grid(True)
        
        schedule_params = f"_schedule{args.schedule}"
        if args.schedule == 'a':
            schedule_params += f"_a{args.a}"
        plot_filename = f'svm_convergence_C{C:.3f}_gamma{args.gamma0}{schedule_params}_{timestamp}.png'
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()
        
        print(f"\nResults for C = {C:.3f}")
        print(f"Training Error: {train_error:.4f}")
        print(f"Test Error: {test_error:.4f}")
    
    results_filename = f'svm_results_schedule{args.schedule}_gamma{args.gamma0}_{timestamp}.txt'
    with open(os.path.join(output_dir, results_filename), 'w') as f:
        f.write(f"SVM Results (schedule={args.schedule}, gamma_0={args.gamma0})\n")
        if args.schedule == 'a':
            f.write(f"Schedule parameter a={args.a}\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            f.write(f"C = {result['C']:.3f}\n")
            f.write(f"Training Error: {result['train_error']:.4f}\n")
            f.write(f"Test Error: {result['test_error']:.4f}\n")
            f.write("-" * 30 + "\n")

if __name__ == "__main__":
    main()
import argparse
from perceptron import Perceptron
from perceptron.utils import load_and_preprocess_data
import numpy as np
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run the Perceptron model on bank-note authentication data.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--standard', action='store_true', help='Run standard perceptron')
    group.add_argument('--voted', action='store_true', help='Run voted perceptron')
    args = parser.parse_args()

    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    # Train Perceptron
    perceptron = Perceptron(learning_rate=0.1, max_epochs=10)

    if args.standard:
        perceptron.fit(X_train, y_train, voted=False)
        y_pred = perceptron.predict_standard(X_test)

        print(f'Learned Weights: {perceptron.weights}')
        print(f'Bias: {perceptron.bias:.2f}')

        average_error = np.mean(y_pred != y_test)
        print(f'\nAverage Test Error: {average_error * 100:.2f}%')

    elif args.voted:
        perceptron.fit(X_train, y_train, voted=True)
        y_pred = perceptron.predict_voted(X_test)

        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'voted_perceptron_results_{timestamp}.txt'
        filepath = os.path.join(output_dir, filename)

        weights_and_counts = perceptron.get_weights_and_counts()
        average_error = np.mean(y_pred != y_test)
        
        with open(filepath, 'w') as f:
            f.write("Voted Perceptron Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total number of distinct weight vectors: {len(weights_and_counts)}\n\n")
            
            total_predictions = sum(count for _, _, count in weights_and_counts)
            f.write(f"Total correct predictions across all vectors: {total_predictions}\n\n")
            
            f.write("Weight Vectors and their Counts:\n")
            f.write("-" * 50 + "\n")
            
            for i, (weights, bias, count) in enumerate(weights_and_counts):
                f.write(f"\nWeight Vector {i+1}:\n")
                f.write(f"Weights: {weights}\n")
                f.write(f"Bias: {bias:.2f}\n")
                f.write(f"Count (correct predictions): {count}\n")
                f.write(f"Percentage of total predictions: {(count/total_predictions)*100:.2f}%\n")
                f.write("-" * 30 + "\n")
            
            f.write(f"\nAverage Test Error: {average_error * 100:.2f}%\n")

        print("\nVoted Perceptron Results:")
        print("Weight Vectors and their Counts:")
        for i, (weights, bias, count) in enumerate(perceptron.get_weights_and_counts()):
            print(f"\nWeight Vector {i+1}:")
            print(f"Weights: {weights}")
            print(f"Bias: {bias:.2f}")
            print(f"Count: {count}")

        print(f'\nAverage Test Error: {average_error * 100:.2f}%')
    

if __name__ == "__main__":
    main()
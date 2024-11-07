# main.py
import argparse
# from perceptron.perceptron import Perceptron
# from perceptron.utils import load_and_preprocess_data
from perceptron import Perceptron
from perceptron.utils import load_and_preprocess_data
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Run the Perceptron model on bank-note authentication data.')

    # parser.add_argument('--data', type=str, required=True, help='Path to the data CSV file.')
    # args = parser.parse_args()

    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    # Train Perceptron
    perceptron = Perceptron(learning_rate=0.1, max_epochs=10)
    perceptron.fit(X_train, y_train)

    # Predict on test data
    y_pred = perceptron.predict(X_test)

    # Calculate average prediction error
    average_error = np.mean(y_pred != y_test)

    # Display learned weight vector and average prediction error
    learned_weights = perceptron.weights
    bias = perceptron.bias

    print(f'Learned Weights: {learned_weights}')
    print(f'Bias: {bias:.2f}')
    print(f'Average Prediction Error: {average_error * 100:.2f}%')

if __name__ == "__main__":
    main()
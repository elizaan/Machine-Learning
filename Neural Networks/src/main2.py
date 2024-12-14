# src/main2.py
import numpy as np
import matplotlib.pyplot as plt
from neural_networks.neural_network import NeuralNetwork

def load_data(filename):

    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)  # Ensure y has the shape (n_samples, 1)
    return X, y

def train_sgd(nn, X_train, y_train, X_test, y_test, width, gamma_0, d, epochs, batch_size):
    n_samples = X_train.shape[0]
    update_count = 0
    losses = []  # Track loss for each update

    for epoch in range(epochs):
        # Shuffle the training data
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Mini-batch SGD
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            activations, z_values = nn.forward_pass(X_batch)

            # Compute gradients
            gradients = nn.backward_pass(X_batch, y_batch, activations, z_values)

            # Update weights using SGD
            gamma_t = gamma_0 / (1 + (gamma_0 / d) * update_count)  # Learning rate schedule
            for layer in range(1, nn.num_layers):
                nn.weights[layer] -= gamma_t * gradients[f'W{layer}']

            # Compute loss and track
            loss = 0.5 * np.mean((activations[nn.num_layers - 1] - y_batch) ** 2)
            losses.append(loss)

            update_count += 1

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    # Plot objective function curve
    plt.plot(losses)
    plt.title(f'Objective Function Curve (Width={width})')
    plt.xlabel('Update Count')
    plt.ylabel('Objective Function Value')
    plt.show()

    # Evaluate training and test error
    train_error = evaluate(nn, X_train, y_train)
    test_error = evaluate(nn, X_test, y_test)

    print(f"Width={width}, Training Error={train_error:.4f}, Test Error={test_error:.4f}")

    return train_error, test_error

def evaluate(nn, X, y):
    activations, _ = nn.forward_pass(X)
    predictions = np.round(activations[nn.num_layers - 1])  # Threshold at 0.5
    accuracy = np.mean(predictions == y)
    return 1 - accuracy  # Return error rate

def main():
    # Load training and test data
    X_train, y_train = load_data('./data/train.csv')  # Replace with actual training data path
    X_test, y_test = load_data('./data/test.csv')    # Replace with actual test data path

    # Hyperparameters
    gamma_0 = 0.1
    d = 100
    epochs = 50
    batch_size = 32
    widths = [5, 10, 25, 50, 100]

    results = []
    for width in widths:
        print(f"\nTraining with hidden layer width={width}")
        
        # Update layer sizes
        layer_sizes = [X_train.shape[1], width, width, 1]
        nn = NeuralNetwork(layer_sizes)

        # Train with SGD
        train_error, test_error = train_sgd(nn, X_train, y_train, X_test, y_test,
                                            width, gamma_0, d, epochs, batch_size)

        results.append((width, train_error, test_error))

    # Print final results
    print("\nFinal Results:")
    for width, train_error, test_error in results:
        print(f"Width={width}, Train Error={train_error:.4f}, Test Error={test_error:.4f}")

if __name__ == "__main__":
    main()

import numpy as np
import os

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights = {}
        self.biases = {}
        
    def load_weights_from_file(self, weight_file):
        # Get absolute path to weights file
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        weight_path = os.path.join(current_dir, 'data', weight_file)
        
        weights_dict = {}
        with open(weight_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                layer = int(parts[0])
                i = int(parts[1][1])
                j = int(parts[1][2])
                value = float(parts[2])
                key = f'w_{i}{j}^{layer}'
                weights_dict[key] = value
                
        # Set up weights matrices
        for layer in range(1, self.num_layers):
            prev_size = self.layer_sizes[layer-1]
            curr_size = self.layer_sizes[layer]
            W = np.zeros((prev_size + 1, curr_size))
            # W = np.zeros((prev_size, curr_size))
            
            for i in range(prev_size):
                for j in range(curr_size):
                    key = f'w_{i}{j}^{layer}'
                    if key in weights_dict:
                        W[i,j] = weights_dict[key]
            
            self.weights[layer] = W
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward_pass(self, X):
        """
        Forward pass with proper bias handling.
        """
        activations = {0: X}  # Input layer
        z_values = {}

        for layer in range(1, self.num_layers):
            # Get weights for the current layer
            W = self.weights[layer]
            
            activations[layer-1] = np.column_stack([np.ones(activations[layer-1].shape[0]), activations[layer-1]])

            # Perform dot product and apply activation function
            z = np.dot(activations[layer-1], W)
            z_values[layer] = z
            activations[layer] = self.sigmoid(z)
            
        return activations, z_values

    def backward_pass(self, X, y, activations, z_values):
        
        m = X.shape[0]  # Number of samples
        gradients = {}  # Dictionary to store gradients

        # Compute delta for output layer
        delta = (activations[self.num_layers - 1] - y) * \
                self.sigmoid_derivative(z_values[self.num_layers - 1])

        # Compute gradients for weights (output layer)
        gradients[f'W{self.num_layers - 1}'] = np.dot(activations[self.num_layers - 2].T, delta) / m

        # Backpropagate through the hidden layers
        for layer in range(self.num_layers - 2, 0, -1):
            # Remove bias from weight matrix before propagating delta
            delta = np.dot(delta, self.weights[layer + 1][1:, :].T) * \
                    self.sigmoid_derivative(z_values[layer])

            # Compute gradients for weights (including bias)
            gradients[f'W{layer}'] = np.dot(activations[layer - 1].T, delta) / m

        return gradients


def validate_gradients():
    """Compare computed gradients with manual calculations"""
    # Network architecture matching the manual calculation
    layer_sizes = [3, 2, 2, 1]  # [input, hidden1, hidden2, output]
    nn = NeuralNetwork(layer_sizes)
    
    # Load weights from file
    nn.load_weights_from_file('weights.txt')
    
    # Test input from manual calculation
    X = np.array([[1, 1, 1]])  # Input values from manual calculation
    y = np.array([[1]])        # Target value from manual calculation
    
    # Forward pass
    activations, z_values = nn.forward_pass(X)
    
    # Print intermediate values for comparison
    print("Forward Pass Values:")
    for layer in z_values:
        print(f"Layer {layer} z values:", z_values[layer])
        print(f"Layer {layer} activations:", activations[layer])
    
    # Compute gradients
    gradients = nn.backward_pass(X, y, activations, z_values)
    
    # Print gradients for comparison with manual calculations
    print("\nComputed Gradients:")
    for key in sorted(gradients.keys()):
        print(f"\n{key}:")
        print(gradients[key])

if __name__ == "__main__":
    validate_gradients()
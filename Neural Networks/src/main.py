# src/main.py
from neural_networks.utils import create_weights_file
from neural_networks.neural_network import NeuralNetwork
import numpy as np

def main():
    # Create weights file
    create_weights_file()
    
    # Initialize network
    layer_sizes = [3, 2, 2, 1]
    nn = NeuralNetwork(layer_sizes)
    
    # Load weights
    nn.load_weights_from_file('weights.txt')
    
    # Test with example from manual calculations
    X = np.array([[1, 1, 1]])
    y = np.array([[1]])
    
    # Forward pass
    activations, z_values = nn.forward_pass(X)
    
    print("Forward Pass Results:")
    print("==================")
    for layer in z_values:
        print(f"\nLayer {layer}")
        print(f"z values: {z_values[layer]}")
        print(f"activations: {activations[layer]}")
    
    # Compute gradients
    gradients = nn.backward_pass(X, y, activations, z_values)
    
    with open('output/gradients.txt', 'w') as f:
        f.write("Gradients:\n")
        f.write("==========\n\n")
        
        # Layer 1 gradients
        f.write("Layer 1 Gradients:\n")
        for i in range(3):  # input nodes
            for j in range(2):  # hidden nodes
                f.write(f"∂L/∂w_{i}{j+1}¹ = {gradients['W1'][i,j]:.6f}\n")
        
        # Layer 2 gradients
        f.write("\nLayer 2 Gradients:\n")
        for i in range(3):  # input nodes
            for j in range(2):  # hidden nodes
                f.write(f"∂L/∂w_{i}{j+1}² = {gradients['W2'][i,j]:.6f}\n")
        
        # Layer 3 gradients
        f.write("\nLayer 3 Gradients:\n")
        for i in range(3):  # input nodes
            f.write(f"∂L/∂w_{i}1³ = {gradients['W3'][i,0]:.6f}\n")

if __name__ == "__main__":
    main()
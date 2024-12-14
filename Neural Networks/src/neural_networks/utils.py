# src/neural_networks/utils.py
import os

def create_weights_file():
    """Create weights.txt from the given table"""
    # Get the path to the data directory
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print("current", current_dir)
    weights_file = os.path.join(current_dir, 'data', 'weights.txt')
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(current_dir, 'data'), exist_ok=True)
    
    weights = [
        # Layer 1
        (1, "01", -1),
        (1, "02", 1),
        (1, "11", -2),
        (1, "12", 2),
        (1, "21", -3),
        (1, "22", 3),
        # Layer 2
        (2, "01", -1),
        (2, "02", 1),
        (2, "11", -2),
        (2, "12", 2),
        (2, "21", -3),
        (2, "22", 3),
        # Layer 3
        (3, "01", -1),
        (3, "11", 2),
        (3, "21", -1.5)
    ]
    
    with open(weights_file, 'w') as f:
        for layer, indices, value in weights:
            f.write(f"{layer},w{indices},{value}\n")
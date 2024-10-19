import numpy as np

def batch_gradient_descent(X, y, learning_rate=0.01, tolerance=1e-6, max_iterations=1000):
    # Initialize weights to zero
    weights = np.zeros(X.shape[1])
    prev_weights = np.ones(X.shape[1]) 
    iteration = 0
    cost_history = []

    # Perform gradient descent
    while np.linalg.norm(weights - prev_weights) > tolerance and iteration < max_iterations:
        prev_weights = weights.copy()
        predictions = X.dot(weights)
        error = predictions - y

        gradient = (1 / len(y)) * X.T.dot(error)
        weights -= learning_rate * gradient

        cost = (1 / (2 * len(y))) * np.sum(error ** 2)
        cost_history.append(cost)
        iteration += 1

    return weights, cost_history

def stochastic_gradient_descent(X, y, learning_rate=0.01, tolerance=1e-6, max_iterations=10000):
    weights = np.zeros(X.shape[1])
    cost_history = []
    iteration = 0

    # Perform SGD
    while iteration < max_iterations:
        # Shuffle the data before each epoch
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        for i in range(len(y)):
            X_i = X[i, :].reshape(1, -1)  # Single training example
            y_i = y[i]  # Corresponding target value

            prediction = X_i.dot(weights)
            error = prediction - y_i

            gradient = X_i.T.dot(error)

            # Update the weights
            weights -= learning_rate * gradient.flatten()

        # Calculate the cost using the entire training set
        full_predictions = X.dot(weights)
        cost = (1 / (2 * len(y))) * np.sum((full_predictions - y) ** 2)
        cost_history.append(cost)

        # Check convergence if the cost difference is very small
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"Converged at iteration {iteration}")
            break

        iteration += 1

    return weights, cost_history


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow in exp
    return 1 / (1 + np.exp(-z))

class LogisticRegressionMAP:
    def __init__(self, input_dim, prior_variance):
        self.w = np.zeros((input_dim, 1))  # Initialize weights to zero
        self.v = prior_variance  # Prior variance

    def compute_gradient(self, X, y):
        """
        Compute the gradient of the MAP objective function.
        """
        m = X.shape[0]
        predictions = sigmoid(X @ self.w)
        error = y - predictions
        gradient = -(X.T @ error) / m - (1 / self.v) * self.w
        # Gradient clipping
        clip_value = 10
        gradient = np.clip(gradient, -clip_value, clip_value)

        return gradient

    def compute_loss(self, X, y):
        """
        Compute the MAP loss for logging.
        """
        m = X.shape[0]
        predictions = sigmoid(X @ self.w)
        # Stabilize log computation with small constant
        log_likelihood = np.sum(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
        prior_term = -np.sum(self.w**2) / (2 * self.v)
        return -log_likelihood / m + prior_term

    def train_sgd(self, X_train, y_train, X_test, y_test, gamma_0, d, epochs, batch_size):
        """
        Train the logistic regression model using stochastic gradient descent.
        """
        m, n = X_train.shape
        update_count = 0
        losses = []

        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Mini-batch SGD
            for i in range(0, m, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # Compute gradient
                gradient = self.compute_gradient(X_batch, y_batch)

                # Update learning rate
                gamma_t = gamma_0 / (1 + (gamma_0 / d) * update_count)

                # Update weights
                self.w -= gamma_t * gradient

                # Log loss
                loss = self.compute_loss(X_batch, y_batch)
                losses.append(loss)

                update_count += 1

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        # Plot objective function curve
        plt.plot(losses)
        plt.title(f'Objective Function Curve (v={self.v})')
        plt.xlabel('Update Count')
        plt.ylabel('Objective Function Value')
        plt.show()

        # Evaluate errors
        train_error = self.evaluate(X_train, y_train)
        test_error = self.evaluate(X_test, y_test)

        print(f"Prior Variance={self.v}, Train Error={train_error:.4f}, Test Error={test_error:.4f}")

        return train_error, test_error

    def evaluate(self, X, y):
        """
        Evaluate the model on a dataset.
        """
        predictions = sigmoid(X @ self.w) >= 0.5
        error = np.mean(predictions != y)
        return error


def load_data(filename):
    """
    Load data from the given CSV file.
    """
    data = np.genfromtxt(filename, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    # Add intercept term
    X = np.column_stack((np.ones(X.shape[0]), X))
    return X, y


def main():
    # Load the bank-note dataset
    X_train, y_train = load_data('./data/train.csv')
    X_test, y_test = load_data('./data/test.csv')

    # Hyperparameters
    gamma_0 = 0.01
    d = 10
    epochs = 100
    batch_size = 32
    variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

    results = []

    for v in variances:
        print(f"\nTraining with prior variance={v}")
        model = LogisticRegressionMAP(input_dim=X_train.shape[1], prior_variance=v)
        train_error, test_error = model.train_sgd(X_train, y_train, X_test, y_test,
                                                  gamma_0=gamma_0, d=d, epochs=epochs, batch_size=batch_size)
        results.append((v, train_error, test_error))

    # Print final results
    print("\nFinal Results:")
    for v, train_error, test_error in results:
        print(f"Variance={v}, Train Error={train_error:.4f}, Test Error={test_error:.4f}")


if __name__ == "__main__":
    main()

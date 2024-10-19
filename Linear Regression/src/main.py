from algorithms.gradient_descent import batch_gradient_descent, stochastic_gradient_descent
from algorithms.utils import load_data
import matplotlib.pyplot as plt
import numpy as np

def calculate_analytical_optimal(X, y):
    weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return weights

train_X, train_y = load_data('data/concrete/train.csv')
test_X, test_y = load_data('data/concrete/test.csv')

optimal_weights = calculate_analytical_optimal(train_X, train_y)
print("Optimal weight vector (analytical solution):")
print(optimal_weights)

# batch gradient descent
learning_rates = [1, 0.5, 0.25, 0.125, 0.0625]
cost_history = []
learning_rate_history = []

for lr in learning_rates:
    weights, cost_history = batch_gradient_descent(train_X, train_y, learning_rate=lr)
    learning_rate_history.extend([lr] * len(cost_history))

print("Length of cost history: ", len(cost_history))
# print("Learning rate history: ", learning_rate_history)
print("Weights", weights)

plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.savefig('./figures/cost_convergence.png')
print("Learning rates used over iterations:")
print(learning_rate_history[:1000])
plt.show()


# test_predictions = test_X.dot(optimal_weights)
# test_error = (1 / (2 * len(test_y))) * np.sum((test_predictions - test_y) ** 2)
# print(f'Test Set Error: {test_error}')

# This values are for stochastic gradient descent
learning_rate = 0.01  # Start with a reasonable initial learning rate
tolerance = 1e-6
max_iterations = 10000
# Stochastic Gradient Descent
weights_s, cost_history_s = stochastic_gradient_descent(
    train_X, train_y, learning_rate=learning_rate, tolerance=tolerance, max_iterations=max_iterations
)
# Report the learned weight vector and learning rate
print("Learned weight vector:")
print(weights_s)
print(f"Chosen learning rate: {learning_rate}")

plt.plot(cost_history_s)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('SGD Cost Function Convergence')
plt.savefig('./figures/sgd_cost_convergence.png')
plt.show()

test_predictions_s = test_X.dot(weights_s)
test_cost_s = (1 / (2 * len(test_y))) * np.sum((test_predictions_s - test_y) ** 2)
print(f'Test Set Cost: {test_cost_s}')

# compare results with analytical solution

# Compare the results
print("\nComparing the weight vectors:")
print("Difference between Analytical and BGD:", np.linalg.norm(optimal_weights - weights))
print("Difference between Analytical and SGD:", np.linalg.norm(optimal_weights - weights_s))

def calculate_cost(X, y, weights):
    predictions = X.dot(weights)
    cost = (1 / (2 * len(y))) * np.sum((predictions - y) ** 2)
    return cost

test_cost_analytical = calculate_cost(test_X, test_y, optimal_weights)
test_cost_bgd = calculate_cost(test_X, test_y, weights)
test_cost_sgd = calculate_cost(test_X, test_y, weights_s)

print("\nCost on the test set:")
print(f"Analytical solution cost: {test_cost_analytical}")
print(f"Batch Gradient Descent cost: {test_cost_bgd}")
print(f"Stochastic Gradient Descent cost: {test_cost_sgd}")
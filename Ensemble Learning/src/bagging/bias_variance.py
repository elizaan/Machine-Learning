import numpy as np
from bagging.bagging import BaggedTrees
from decision_tree.decision_tree import predict
from decision_tree.feature_def import get_feature_definitions

def compute_bias_variance(predictions, true_label):
    """
    Compute the bias squared and variance for a set of predictions.
    
    Args:
        predictions (np.array): Array of predictions.
        true_label (float): The ground-truth label.
    
    Returns:
        bias_squared (float): The squared bias term.
        variance (float): The variance term.
    """
    mean_prediction = np.mean(predictions)
    bias_squared = (mean_prediction - true_label) ** 2
    variance = np.var(predictions)
    return bias_squared, variance

def run_bias_variance(data, test_data, n_iterations=100, n_trees=500, sample_size=1000, max_depth=None, info_gain='entropy'):
    """
    Run the bias-variance decomposition experiment.
    
    Args:
        data (np.array): The training data.
        test_data (np.array): The test data.
        n_iterations (int): Number of iterations for sampling and training.
        n_trees (int): Number of trees in the bagged ensemble.
        sample_size (int): Number of examples to sample in each iteration.
        max_depth (int): Maximum depth for individual trees.
        info_gain (str): Information gain criterion for decision trees.
    
    Returns:
        single_tree_results (dict): Dictionary containing bias, variance, and error for single trees.
        bagged_tree_results (dict): Dictionary containing bias, variance, and error for bagged trees.
    """
    # Initialize arrays to collect predictions
    single_tree_predictions = []
    bagged_tree_predictions = []

    # Run the experiment for n_iterations
    for _ in range(n_iterations):
        # Step 1: Sample the training data
        sampled_indices = np.random.choice(len(data), size=sample_size, replace=False)
        sampled_data = data[sampled_indices]

        # Step 2: Train the bagged trees model on the sampled data
        bagged_model = BaggedTrees(num_trees=n_trees, max_depth=max_depth)
        Feature, Column, _, Numeric_Attributes = get_feature_definitions('bank')  # Change dataset as needed
        bagged_model.fit(sampled_data, Feature, Column, Numeric_Attributes, info_gain=info_gain)

        # Store predictions for the bagged model
        bagged_predictions_for_iteration = []
        for test_instance in test_data:
            prediction = bagged_model.bagged_predict(test_instance[:-1], Column, Numeric_Attributes)
            bagged_predictions_for_iteration.append(prediction)
        bagged_tree_predictions.append(bagged_predictions_for_iteration)

        # Store predictions for the first tree in the bagged model
        first_tree = bagged_model.trees[0]
        single_tree_predictions_for_iteration = []
        for test_instance in test_data:
            prediction = predict(test_instance[:-1], first_tree, Column, Numeric_Attributes)
            single_tree_predictions_for_iteration.append(prediction)
        single_tree_predictions.append(single_tree_predictions_for_iteration)

    # Convert to numpy arrays for easier calculations
    single_tree_predictions = np.array(single_tree_predictions)
    bagged_tree_predictions = np.array(bagged_tree_predictions)

    # Compute bias, variance, and error for each test example
    single_tree_bias_squared = []
    single_tree_variance = []
    single_tree_error = []

    bagged_tree_bias_squared = []
    bagged_tree_variance = []
    bagged_tree_error = []

    for i in range(len(test_data)):
        true_label = test_data[i, -1]

        # Bias and variance for single tree predictions
        bias_sq, var = compute_bias_variance(single_tree_predictions[:, i], true_label)
        single_tree_bias_squared.append(bias_sq)
        single_tree_variance.append(var)
        single_tree_error.append(bias_sq + var)

        # Bias and variance for bagged tree predictions
        bias_sq, var = compute_bias_variance(bagged_tree_predictions[:, i], true_label)
        bagged_tree_bias_squared.append(bias_sq)
        bagged_tree_variance.append(var)
        bagged_tree_error.append(bias_sq + var)

        if i % 15 == 0:
            print(f"Number of iteration: {i}")

    # Average bias, variance, and error across all test examples
    single_tree_results = {
        'bias_squared': np.mean(single_tree_bias_squared),
        'variance': np.mean(single_tree_variance),
        'error': np.mean(single_tree_error)
    }

    bagged_tree_results = {
        'bias_squared': np.mean(bagged_tree_bias_squared),
        'variance': np.mean(bagged_tree_variance),
        'error': np.mean(bagged_tree_error)
    }

    return single_tree_results, bagged_tree_results

# def main():
#     # Load data
#     train_data, test_data = load_and_preprocess_data('bank')  # Change dataset as needed

#     # Run the bias-variance experiment
#     single_tree_results, bagged_tree_results = run_bias_variance_experiment(
#         train_data, test_data, n_iterations=100, n_trees=500, sample_size=1000, max_depth=None, info_gain='entropy'
#     )

#     # Display results
#     print("Single Tree Results:")
#     print(f"Bias^2: {single_tree_results['bias_squared']:.4f}")
#     print(f"Variance: {single_tree_results['variance']:.4f}")
#     print(f"Error: {single_tree_results['error']:.4f}")

#     print("\nBagged Trees Results:")
#     print(f"Bias^2: {bagged_tree_results['bias_squared']:.4f}")
#     print(f"Variance: {bagged_tree_results['variance']:.4f}")
#     print(f"Error: {bagged_tree_results['error']:.4f}")

# if __name__ == "__main__":
#     main()

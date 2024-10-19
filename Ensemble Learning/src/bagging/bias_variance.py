import numpy as np
from bagging.bagging import BaggedTrees
from bagging.random_forest import RandomForest
from decision_tree.decision_tree import predict
from decision_tree.feature_def import get_feature_definitions

def compute_bias_variance(predictions, true_label):
    # Manually map 'yes' to 1 and 'no' to 0
    mapping = {'yes': 1, 'no': 0}
    if isinstance(predictions[0], str):
        predictions = np.array([mapping[p] for p in predictions])
        true_label = mapping[true_label]
    
    predictions = np.array(predictions, dtype=float)
    
    mean_prediction = np.mean(predictions)
    bias_squared = (mean_prediction - true_label) ** 2
    variance = np.var(predictions)
    return bias_squared, variance

def run_bias_variance(data, test_data, n_iterations=100, n_trees=500, sample_size=1000, max_depth=None, subset_size=2, info_gain='entropy', method='bagging'):
    """
    Run the bias-variance decomposition experiment for bagging or random forest.
    
    Args:
        data (np.array): The training data.
        test_data (np.array): The test data.
        n_iterations (int): Number of iterations for sampling and training.
        n_trees (int): Number of trees in the ensemble.
        sample_size (int): Number of examples to sample in each iteration.
        max_depth (int): Maximum depth for individual trees.
        subset_size (int): Size of the feature subset for each split (used for random forest).
        info_gain (str): Information gain criterion for decision trees.
        method (str): 'bagging' or 'rf' (random forest).
    
    Returns:
        single_tree_results (dict): Dictionary containing bias, variance, and error for single trees.
        ensemble_results (dict): Dictionary containing bias, variance, and error for the ensemble.
    """
    # Initialize arrays to collect predictions
    single_tree_predictions = []
    ensemble_predictions = []

    # Run the experiment for n_iterations
    for _ in range(n_iterations):
        # Step 1: Sample the training data
        sampled_indices = np.random.choice(len(data), size=sample_size, replace=False)
        sampled_data = data[sampled_indices]

        # Step 2: Train the ensemble model on the sampled data
        Feature, Column, _, Numeric_Attributes = get_feature_definitions('bank')  # Change dataset as needed
        
        if method == 'bagging':
            model = BaggedTrees(num_trees=n_trees, max_depth=max_depth)
            model.fit(sampled_data, Feature, Column, Numeric_Attributes, info_gain=info_gain)
        elif method == 'rf':
            model = RandomForest(num_trees=n_trees, max_depth=max_depth, subset_size=subset_size)
            model.fit(sampled_data, Feature, Column, Numeric_Attributes, info_gain=info_gain)
        else:
            raise ValueError("Invalid method specified. Use 'bagging' or 'rf'.")

        # Store predictions for the ensemble model
        ensemble_predictions_for_iteration = []
        for test_instance in test_data:
            prediction = model.bagged_predict(test_instance[:-1], Column, Numeric_Attributes) if method == 'bagging' else model.predict(test_instance[:-1], Column, Numeric_Attributes)
            ensemble_predictions_for_iteration.append(prediction)
        ensemble_predictions.append(ensemble_predictions_for_iteration)

        # Store predictions for the first tree in the ensemble model
        first_tree = model.trees[0]
        single_tree_predictions_for_iteration = []
        for test_instance in test_data:
            prediction = predict(test_instance[:-1], first_tree, Column, Numeric_Attributes)
            single_tree_predictions_for_iteration.append(prediction)
        single_tree_predictions.append(single_tree_predictions_for_iteration)

    # Convert to numpy arrays for easier calculations
    single_tree_predictions = np.array(single_tree_predictions)
    ensemble_predictions = np.array(ensemble_predictions)

    # Compute bias, variance, and error for each test example
    single_tree_bias_squared = []
    single_tree_variance = []
    single_tree_error = []

    ensemble_bias_squared = []
    ensemble_variance = []
    ensemble_error = []

    for i in range(len(test_data)):
        true_label = test_data[i, -1]

        # Bias and variance for single tree predictions
        bias_sq, var = compute_bias_variance(single_tree_predictions[:, i], true_label)
        single_tree_bias_squared.append(bias_sq)
        single_tree_variance.append(var)
        single_tree_error.append(bias_sq + var)

        # Bias and variance for ensemble predictions
        bias_sq, var = compute_bias_variance(ensemble_predictions[:, i], true_label)
        ensemble_bias_squared.append(bias_sq)
        ensemble_variance.append(var)
        ensemble_error.append(bias_sq + var)

    # Average bias, variance, and error across all test examples
    single_tree_results = {
        'bias_squared': np.mean(single_tree_bias_squared),
        'variance': np.mean(single_tree_variance),
        'error': np.mean(single_tree_error)
    }

    ensemble_results = {
        'bias_squared': np.mean(ensemble_bias_squared),
        'variance': np.mean(ensemble_variance),
        'error': np.mean(ensemble_error)
    }

    return single_tree_results, ensemble_results

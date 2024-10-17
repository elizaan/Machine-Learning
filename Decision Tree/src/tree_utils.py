import numpy as np
from collections import Counter
from feature_def import get_feature_definitions

# Load Feature Definitions Based on Dataset
# def setup_feature_definitions(dataset):
#     global Feature, Column, Label, Numeric_Attributes
#     Feature, Column, Label, Numeric_Attributes = get_feature_definitions(dataset)

# Check if all labels in the subset are the same
def check_same_label(subset_idx, train_data):
    first_label = train_data[subset_idx[0]][-1]
    subset_size = len(subset_idx)
    for i in range(1, subset_size):
        if train_data[subset_idx[i]][-1] != first_label:
            return first_label, False
    return first_label, True

# Find the most common label in the subset
def find_most_common_label(subset_idx, train_data):
    labels = [train_data[idx][-1] for idx in subset_idx]
    return Counter(labels).most_common(1)[0][0]

# Separate data based on attribute value
def data_separate(subset_idx, attribute, train_data, Column, Numeric_Attributes, Feature):

    # print("attribute: ", attribute)
    # print("column5: ", Column)
    attribute_index = Column.index(attribute)
    
    if attribute in Numeric_Attributes:
        # For numeric attributes, values have been converted to binary (0 or 1)
        separated_data = [[], []]
        for idx in subset_idx:
            value = int(train_data[idx][attribute_index])
            separated_data[value].append(idx)
    else:
        attribute_values = Feature[attribute]
        separated_data = [[] for _ in attribute_values]
        
        for idx in subset_idx:
            value = train_data[idx][attribute_index]
            # print(f"Processing attribute '{attribute}', value: '{value}', attribute values: {attribute_values}")
            value_index = attribute_values.index(value)
            separated_data[value_index].append(idx)
    
    return [np.array(subset) for subset in separated_data]

# Compute entropy for the given subset
def compute_entropy(subset_idx, train_data):
    if len(subset_idx) == 0:
        return 0
    
    labels = [train_data[idx][-1] for idx in subset_idx]
    label_counts = Counter(labels)
    
    probabilities = [count / len(subset_idx) for count in label_counts.values()]
    return -sum(p * np.log2(p) for p in probabilities)

# Compute majority error for the given subset
def compute_majority_error(subset_idx, train_data):
    if len(subset_idx) == 0:
        return 0
    
    labels = [train_data[idx][-1] for idx in subset_idx]
    label_counts = Counter(labels)
    
    majority_count = max(label_counts.values())
    return 1 - (majority_count / len(subset_idx))

# Compute Gini index for the given subset
def compute_gini_index(subset_idx, train_data):
    if len(subset_idx) == 0:
        return 0
    
    labels = [train_data[idx][-1] for idx in subset_idx]
    label_counts = Counter(labels)
    
    probabilities = [(count / len(subset_idx))**2 for count in label_counts.values()]
    return 1 - sum(probabilities)

# Compute information gain for an attribute
def compute_information_gain(subset_idx, attribute, train_data, Column, Numeric_Attributes, Feature, option='entropy'):
    # print("column4: ", Column)

    separated_subsets = data_separate(subset_idx, attribute, train_data, Column, Numeric_Attributes, Feature)
    subset_size = len(subset_idx)
    
    if option == 'entropy':
        compute_func = compute_entropy
    elif option == 'majority_error':
        compute_func = compute_majority_error
    elif option == 'gini_index':
        compute_func = compute_gini_index
    else:
        raise ValueError("Invalid option. Choose 'entropy', 'majority_error', or 'gini_index'.")
    
    initial_value = compute_func(subset_idx, train_data)
    weighted_sum = sum(
        len(subset) / subset_size * compute_func(subset, train_data)
        for subset in separated_subsets
    )
    return initial_value - weighted_sum

# Find the best attribute to split on
def best_split(subset_idx, attributes, train_data, Column, Numeric_Attributes, Feature, option='entropy'):

    # print("column3: ", Column)
    information_gains = [
        compute_information_gain(subset_idx, attr, train_data, Column, Numeric_Attributes, Feature, option)
        for attr in attributes
    ]
    return attributes[np.argmax(information_gains)]

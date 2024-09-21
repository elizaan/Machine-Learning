import numpy as np
from collections import Counter
from feature_def import Feature, Column, Label, Numeric_Attributes

def check_same_label(subset_idx, train_data):
    first_label = train_data[subset_idx[0]][-1]
    subset_size = len(subset_idx)
    for i in range(1, subset_size):
        if train_data[subset_idx[i]][-1] != first_label:
            return first_label, False
    return first_label, True

def find_most_common_label(subset_idx, train_data):
    labels = [train_data[idx][-1] for idx in subset_idx]
    return Counter(labels).most_common(1)[0][0]

# def data_separate(subset_idx, attribute, train_data):
#     attribute_values = Feature[attribute]
#     attribute_index = Column.index(attribute)
    
#     separated_data = [[] for _ in attribute_values]
    
#     for idx in subset_idx:
#         value = train_data[idx][attribute_index]
#         value_index = attribute_values.index(value)
#         separated_data[value_index].append(idx)
    
#     return [np.array(subset) for subset in separated_data]

def data_separate(subset_idx, attribute, train_data):
    attribute_index = Column.index(attribute)
    
    if attribute in Numeric_Attributes:
        # For numeric attributes, we've converted them to binary (0 or 1)
        separated_data = [[], []]
        for idx in subset_idx:
            value = int(train_data[idx][attribute_index])
            separated_data[value].append(idx)
    else:
        attribute_values = Feature[attribute]
        separated_data = [[] for _ in attribute_values]
        for idx in subset_idx:
            value = train_data[idx][attribute_index]
            value_index = attribute_values.index(value)
            separated_data[value_index].append(idx)
    
    return [np.array(subset) for subset in separated_data]

def compute_entropy(subset_idx, train_data):
    if len(subset_idx) == 0:
        return 0
    
    labels = [train_data[idx][-1] for idx in subset_idx]
    label_counts = Counter(labels)
    
    probabilities = [count / len(subset_idx) for count in label_counts.values()]
    return -sum(p * np.log2(p) for p in probabilities)

# def compute_information_gain(subset_idx, attribute, train_data):
#     separated_subsets = data_separate(subset_idx, attribute, train_data)
#     subset_size = len(subset_idx)
    
#     initial_entropy = compute_entropy(subset_idx, train_data)
#     weighted_entropy_sum = sum(
#         len(subset) / subset_size * compute_entropy(subset, train_data)
#         for subset in separated_subsets
#     )
#     return initial_entropy - weighted_entropy_sum

def compute_majority_error(subset_idx, train_data):
    if len(subset_idx) == 0:
        return 0
    
    labels = [train_data[idx][-1] for idx in subset_idx]
    label_counts = Counter(labels)
    
    majority_count = max(label_counts.values())
    return 1 - (majority_count / len(subset_idx))

def compute_gini_index(subset_idx, train_data):
    if len(subset_idx) == 0:
        return 0
    
    labels = [train_data[idx][-1] for idx in subset_idx]
    label_counts = Counter(labels)
    
    probabilities = [(count / len(subset_idx))**2 for count in label_counts.values()]
    return 1 - sum(probabilities)

def compute_information_gain(subset_idx, attribute, train_data, option='entropy'):
    separated_subsets = data_separate(subset_idx, attribute, train_data)
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

def best_split(subset_idx, attributes, train_data, option='entropy'):
    information_gains = [
        compute_information_gain(subset_idx, attr, train_data, option)
        for attr in attributes
    ]
    return attributes[np.argmax(information_gains)]
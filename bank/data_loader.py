import csv
import numpy as np
from feature_def import Feature, Column, Label

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data.append(row)
    return np.array(data)

def preprocess_data(data, numeric_attributes, handle_unknown='as_value'):
    processed_data = data.copy()
    
    for i, attr in enumerate(Column):
        if attr in numeric_attributes:
            median = np.median([float(x) for x in processed_data[:, i] if x != 'unknown'])
            processed_data[:, i] = np.where(processed_data[:, i] != 'unknown',
                                            (processed_data[:, i].astype(float) > median).astype(int),
                                            processed_data[:, i])
        
        if handle_unknown == 'complete':
            values = [x for x in processed_data[:, i] if x != 'unknown']
            most_common = max(set(values), key=values.count)
            processed_data[:, i] = np.where(processed_data[:, i] == 'unknown', most_common, processed_data[:, i])
    
    return processed_data

def load_and_preprocess_data(train_path, test_path, numeric_attributes, handle_unknown='as_value'):
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    processed_train_data = preprocess_data(train_data, numeric_attributes, handle_unknown)
    processed_test_data = preprocess_data(test_data, numeric_attributes, handle_unknown)
    
    return processed_train_data, processed_test_data
import csv
import numpy as np
import argparse
import os
# from feature_def import get_feature_definitions, Feature, Column, Label
from feature_def import get_feature_definitions

# Function to load CSV data
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data.append(row)
    return np.array(data)

# Function to load car data (simple version)
def load_car_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            data.append(terms)
    return np.array(data)

# Preprocess bank data
def preprocess_data(data, numeric_attributes, handle_unknown='as_value'):
    processed_data = data.copy()
    
    _, Column, _, _ = get_feature_definitions("bank")

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

# Load and preprocess data for both bank and car datasets
def load_and_preprocess_data(dataset, handle_unknown='as_value'):
    base_path = os.path.join(os.path.dirname(__file__), '../data')
    
    if dataset == 'bank':
        train_path = os.path.join(base_path, 'bank/train.csv')
        test_path = os.path.join(base_path, 'bank/test.csv')
        _, Column, _, _ = get_feature_definitions(dataset)

        numeric_attributes = [Column[0], Column[5], Column[9], Column[11], Column[12], Column[13], Column[14]]
        
        train_data = load_data(train_path)
        test_data = load_data(test_path)
        
        processed_train_data = preprocess_data(train_data, numeric_attributes, handle_unknown)
        processed_test_data = preprocess_data(test_data, numeric_attributes, handle_unknown)

        
    elif dataset == 'car':
        train_path = os.path.join(base_path, 'car/train.csv')
        test_path = os.path.join(base_path, 'car/test.csv')
        
        processed_train_data = load_car_data(train_path)
        processed_test_data = load_car_data(test_path)
    else:
        raise ValueError("Invalid dataset. Please use 'bank' or 'car'.")
    
    return processed_train_data, processed_test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Specify the dataset to load: 'bank' or 'car'")
    args = parser.parse_args()
    
    # Load and preprocess the data
    train_data, test_data = load_and_preprocess_data(args.data)
    
    # Print confirmation
    # print(f"Loaded and preprocessed {args.data} dataset successfully.")
    # print("Train Data Sample:")
    # print(train_data[:5])
    # print("Test Data Sample:")
    # print(test_data[:5])

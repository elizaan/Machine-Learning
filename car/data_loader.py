def load_train_data(file_path='train.csv'):
    return load_data(file_path)

def load_test_data(file_path='test.csv'):
    return load_data(file_path)

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            data.append(terms)
    return data

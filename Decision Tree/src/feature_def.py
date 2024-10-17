# Feature definitions for both bank and car datasets

def get_feature_definitions(dataset):
    if dataset == 'bank':
        Feature = {
            'age': 'numeric',
            'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                    'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
            'marital': ['married', 'divorced', 'single'],
            'education': ['unknown', 'secondary', 'primary', 'tertiary'],
            'default': ['yes', 'no'],
            'balance': 'numeric',
            'housing': ['yes', 'no'],
            'loan': ['yes', 'no'],
            'contact': ['unknown', 'telephone', 'cellular'],
            'day': 'numeric',
            'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            'duration': 'numeric',
            'campaign': 'numeric',
            'pdays': 'numeric',
            'previous': 'numeric',
            'poutcome': ['unknown', 'other', 'failure', 'success']
        }
        Label = ['yes', 'no']
    elif dataset == 'car':
        Feature = {
            'buying': ['vhigh', 'high', 'med', 'low'],
            'maint': ['vhigh', 'high', 'med', 'low'],
            'doors': ['2', '3', '4', '5more'],
            'persons': ['2', '4', 'more'],
            'lug_boot': ['small', 'med', 'big'],
            'safety': ['low', 'med', 'high']
        }
        Label = ['unacc', 'acc', 'good', 'vgood']
    else:
        raise ValueError("Invalid dataset. Please use 'bank' or 'car'.")
    
    Column = list(Feature.keys())
    Numeric_Attributes = [attr for attr, value in Feature.items() if value == 'numeric']
    
    return Feature, Column, Label, Numeric_Attributes

# Example usage
if __name__ == "__main__":
    dataset = 'bank'  # or 'car'
    Feature, Column, Label, Numeric_Attributes = get_feature_definitions(dataset)
    # print("Feature Definitions:", Feature)
    # print("Columns:", Column)
    # print("Labels:", Label)
    # print("Numeric Attributes:", Numeric_Attributes)

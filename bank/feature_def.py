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

Column = list(Feature.keys())

Label = ['yes', 'no']

Numeric_Attributes = [attr for attr, value in Feature.items() if value == 'numeric']
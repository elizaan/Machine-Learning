# Machine Learning Library

This is a machine learning library developed by Ishrat Jahan Eliza for CS5350/6350 in University of Utah. This repository contains implementations of various machine learning models including Decision Trees, Linear Regression, and Ensemble Learning techniques.
## Repository Structure

The repository is organized as follows:

Machine-Learning/
│
├── Decision Tree/
│   ├── data/
│   │   ├── bank/
│   │   └── car/
│   └── src/
|       ├──decision_tree
|           ├── __init__.py
│           ├── data_loader.py
│           ├── decision_tree.py
│           ├── feature_def.py
│           └── tree_utils.py
|       ├── main.py
│
├── Ensemble Learning/
│
└── Linear Regression/

Machine-Learning/
│
├── Decision Tree/
│   ├── data/
│   │   ├── bank/
│   │   └── car/
│   └── src/
│       ├── decision_tree/
│       │   ├── __init__.py
│       │   ├── data_loader.py
│       │   ├── decision_tree.py
│       │   ├── feature_def.py
│       │   └── tree_utils.py
│       └── main.py
│
└── Ensemble Learning/
    └── src/
        ├── adaboost/
        │   ├── __init__.py
        │   ├── adaboost.py
        │   ├── decision_stump.py
        │   └── utils.py
        └── main.py

- **Decision Tree**: Contains the implementation and relevant data for decision tree learning.
- **Ensemble Learnin**g**: Reserved for future development of ensemble methods such as bagging, boosting, and random forests.
- **Linear Regression**: Reserved for the implementation of linear regression models.

## Installation

In Machine-Learning Folder (parent folder):
- For Linux/macOS
   ```bash
      python3 -m venv env
      source env/bin/activate
      pip install -r requirements.txt
   ```

- For Windows
   ```bash
      python3 -m venv env
      .\env\Scripts\activate
      pip install -r requirements.txt
    ```   


## Commands to run Decision Tree

To run the implementation of the Decision Tree learning algorithm, follow these steps:

1. **Navigate to the Decision Tree Directory**:
   Ensure that you're in the `Decision Tree` directory and navigate to the `src` folder to access the source code:

   ```bash
   cd Machine\ Learning/Decision\ Tree/

2. **Run the Main File**: 
   Use the main.py script to train and test a decision tree model on the dataset of your choice. 

    **Example Command**
    ```bash
    python3 src/main.py --data bank --depth 5 --gi
    ```

    In this example, the command trains a decision tree on the bank dataset with a maximum depth of 5 using the Gini Index as the split criterion.

    **Parameters Description**:
    Dataset (--data): Choose between bank and car datasets. bank for bank and car for car dataset
    Maximum Depth (--depth): Define the maximum depth for the tree. Takes numeroical value.
    Splitting Criteria:
    Gini Index (--gi): Measures the impurity of a node.
    Entropy (--et): Measures the information gain.
    Majority Error (--me): Measures the fraction of misclassified instances.
    
    **Results:**
    Once the command runs, you will receive a tabular output displaying the error on the training and test sets.


## Contact ##
For any questions or suggestions, feel free to contact:

Name: Ishrat Jahan Eliza
Email: ishratjahan.eliza@utah.edu
LinkedIn: 
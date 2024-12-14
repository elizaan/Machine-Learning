# Machine Learning Library

This is a machine learning library developed by Ishrat Jahan Eliza for CS5350/6350 in University of Utah. This repository contains implementations of various machine learning models including Decision Trees, Linear Regression, and Ensemble Learning techniques.
## Repository Structure

The repository is organized as follows:

```bash
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
│   └── src/
│       ├── adaboost/
│       │   ├── __init__.py
│       │   ├── adaboost.py
│       │   ├── decision_stump.py
│       │   └── utils.py
│       ├── bagging/
│       │   ├── __init__.py
│       │   ├── bagging.py
│       │   ├── bias_variance.py
│       │   └── random_forest.py
│       └── main.py
└── Perceptron/
│  └── src/
│   │   ├── main.py
│   │   ├── perceptron/
│   │   │   ├── init.py
│   │   │   ├── perceptron.py
│   │   │   └── utils.py
│   ├── data/
│   │   ├── train.csv
│   │   └── test.csv
└──SVM/
│   ├── data/
│   ├── output/
│   ├── src/
│   │   ├── main.py
│   │   ├── main2.py
│   │   └── svm/
│   │       ├── __init__.py
│   │       ├── svm.py
│   │       ├── dual_svm.py
│   │       └── utils.py
└── Neural Networks/
│   ├── data/
│   │  ├── train.csv
│   │  ├── test.csv
│   │  ├── weights.txt
│   ├── output/
│   ├── src/
│   │   ├── main.py
│   │ 
└── Logostic Regression/
│   ├── data/
│   │  ├── train.csv
│   │  ├── test.csv
│   │  ├── weights.txt
│   ├── lr.py
│   ├── lr2.py

```

- **Decision Tree**: Contains the implementation and relevant data for decision tree learning.
- **Ensemble Learnin**g**: Contains the implementation of adaboost, bagging and random forest.
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
      python -m venv env
      .\env\Scripts\activate
      pip install -r requirements.txt
    ```   
    Sometimes windows do not permit to activate env, for that, open powershell as administration and do run the follwoing command
    ```bash
       Set-ExecutionPolicy RemoteSigned
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
    python src/main.py --data bank --depth 5 --gi
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

## Commands to run Ensemble Learnings

To run the implementation of the Ensemble learning algorithms, follow these steps:

1. **Navigate to the Ensemble Learning Directory**:
   navigate to the `src` folder to access the source code:

   ```bash
   cd Machine\ Learning/Ensemble\ Learning/

2. **Run the Main File**: 
   Use the main.py script to train and test models on the dataset of your choice. 

   - Adaboost:
   ```bash
    python src/main.py --data bank --adaboost
    ```
   - Bagging:
   ```bash
   python src/main.py --data bank --bagging
   ```
   - Random Forest (x can be 2/4/6):
   ```bash
   python src/main.py --data bank --rf --subset_size x
   ```
   - Bias-Variance:
   For Bagging:
   ```bash
   python src/main.py --data bank --bv --bagging
   ```
   For Random Forest (x = 2/4/6):
   ```bash
   python src/main.py --data bank --bv --rf --subset_size x
   ```
   **Results:**
   Once the command runs, you will receive a figure plotting error rates or printing information (bias-variance only).

## Commands to run Linear Regressions

1. **Navigate to the Ensemble Learning Directory**:
   navigate to the `src` folder to access the source code:

   ```bash
   cd Machine\ Learning/Linear\ Regression/

2. **Run the Main File**: 
   Use the main.py script to plot the cost functions for BGD, SGD all at once. 

   ```bash
    python3 src/main.py
    ```

## Commands to run Perceptron

1. **Navigate to the Perceptron Directory**:
   navigate to the `src` folder to access the source code:

   ```bash
   cd Machine\ Learning/Perceptron/

2. **Run the Main File**: 
   Use the main.py script to plot the cost functions for BGD, SGD all at once. 
   
   - Standard Peceptron:
   ```bash
    python src/main.py --standard
    ```
   - Voted Peceptron:
   ```bash
    python src/main.py --voted
    ```
   - Average Peceptron:
   ```bash
    python src/main.py --average
    ```

## Commands to run SVM

1. **Navigate to the SVM Directory**:
   navigate to the `src` folder to access the source code:

   ```bash
   cd Machine\ Learning/SVM/

2. **Run the Main File**: 
   Use the main.py script to plot the cost functions for BGD, SGD all at once. 

   2A
   ```bash
     python src/main.py --schedule a --gamma0 0.1 --a 1.0
    ```
   2B
   ```bash
    python src/main.py --schedule b --gamma0 0.1
    ```
   3a and 3b
   ```bash
    python src/main2.py
    ```

## Commands to run Neural Network

1. **Navigate to the Neural Networks Directory**:
   navigate to the `src` folder to access the source code:

   ```bash
   cd Machine\ Learning/"Neural Networks"/

2. **Run the Main File**:  

   2a
   ```bash
     python src/main.py 
    ```
   2b and 2c
   ```bash
    python src/main2.py 
    ```
## Commands to run Logistic Regression

1. **Navigate to the Neural Networks Directory**:
   navigate to the `src` folder to access the source code:

   ```bash
   cd Machine\ Learning/"Logistic Regression"/

2. **Run the Main File**:  

   3a
   ```bash
     python lr.py 
    ```
   3b
   ```bash
    python lr2.py 

## Contact ##
For any questions or suggestions, feel free to contact:

Name: Ishrat Jahan Eliza
Email: ishratjahan.eliza@utah.edu
LinkedIn: 
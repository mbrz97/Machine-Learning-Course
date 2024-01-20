## Documentation for MLP Classifier with Backpropagation Code
## Project for Machine Learning Course - Persian Gulf University

### Overview
This Python script implements a custom multi-layer perceptron (MLP) classifier for binary classification, specifically demonstrated on a breast cancer dataset. The MLP classifier uses a sigmoid activation function and backpropagation for training.

### Key Components
1. **Imports**: 
    - Standard libraries like `math`, `random`, and `pandas`.
    - `sklearn` modules for data preprocessing and model evaluation.

2. **Classes**:
   - `LogisticActivation`: Defines the sigmoid activation function and its derivative.
   - `Neuron`: Represents a neuron in the network with methods to manage weights and values.
   - `MLPClassifier`: Implements the MLP with methods for training (`Fit`), prediction (`Predict`), and backpropagation (`BackwardOutputLayer` and `BackwardHiddenLayer`).

3. **MLPClassifier**:
   - Initializes a neural network based on specified layer sizes.
   - Includes methods for forward and backward propagation.
   - `Fit` method for training the model over a specified number of epochs.
   - `Predict` method for generating predictions on new data.

4. **Utility Functions**:
   - `box`: A helper function to transform a list into a list of lists.

5. **Data Processing and Model Training**:
   - Loads a breast cancer dataset and preprocesses it (standard scaling, label encoding).
   - Splits data into training and test sets.
   - Trains the MLP classifier on the training data.
   - Predicts and prints model predictions and accuracy on the test set.

### Usage
- Update the data loading path to point to your dataset (currently set to 'breastcancer.csv').
- Optionally adjust the MLP structure by modifying the `layerSizes` parameter in the `MLPClassifier` instantiation.
- Run the script to train the model and evaluate its performance on the test set.

### Notes
- The model's architecture and parameters can be tuned according to the specific requirements of the dataset or the task.
- The script includes print statements for debugging and tracking progress, which can be toggled using the `verbose` parameter in the `MLPClassifier`.
- The model uses a basic mean squared error function for calculating the cost, which is suitable for binary classification tasks.

This script provides a foundational structure for a neural network classifier and can be extended or modified for more complex datasets and tasks.

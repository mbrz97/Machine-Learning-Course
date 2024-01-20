# Custom Perceptron Classifier Documentation

## Overview
This script implements a basic perceptron, a type of artificial neuron used in machine learning for binary classification. The perceptron receives inputs, multiplies them by weights, sums them up, and passes the sum through an activation function to produce an output.

## Functions

### `predict(inputs, weights)`
- **Purpose**: Calculates the output of the perceptron.
- **Parameters**:
  - `inputs`: A list of input values.
  - `weights`: A list of weights corresponding to the inputs.
- **Returns**: The binary classification result (1 or 0).

### `plot(matrix, weights, title)`
- **Purpose**: Visualizes the classification results.
- **Parameters**:
  - `matrix`: A dataset where each row is a list of inputs with the last element being the classification label.
  - `weights`: The weights of the perceptron.
  - `title`: Title of the plot.
- **Functionality**: For 1D and 2D inputs, it plots the data points and decision boundaries.

### `accuracy(matrix, weights)`
- **Purpose**: Calculates the accuracy of the perceptron.
- **Parameters**:
  - `matrix`: The dataset used for testing the perceptron.
  - `weights`: The weights of the perceptron.
- **Returns**: Accuracy as a proportion of correct predictions.

### `train_weights(matrix, weights, nb_epoch, l_rate, do_plot, stop_early, verbose)`
- **Purpose**: Trains the perceptron by adjusting its weights.
- **Parameters**:
  - `matrix`: Training dataset.
  - `weights`: Initial weights.
  - `nb_epoch`: Number of training epochs.
  - `l_rate`: Learning rate.
  - `do_plot`: Boolean to control plotting after each epoch.
  - `stop_early`: Stops training if 100% accuracy is achieved.
  - `verbose`: Prints detailed logs if True.
- **Returns**: The adjusted weights after training.

### `main()`
- **Purpose**: Serves as the entry point for the script.
- **Functionality**: Contains the dataset and calls the training function. It can be modified to test different datasets or perceptron configurations.

## Usage
- Prepare a dataset where each row is a list of input values followed by the classification label.
- Configure the perceptron's initial weights and other parameters in the `main` function.
- Run the script to train the perceptron and visualize the results.

## Dependencies
- Python 3.x
- `matplotlib` for plotting
- `numpy` for numerical operations

## Notes
- The perceptron works for linearly separable datasets.
- The script demonstrates the fundamental concepts of perceptron training and classification but may require modifications for advanced use cases.

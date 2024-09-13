# Neuradriz - Simple Neural Network Module

**Neuradriz** is a pure JavaScript-based simple neural network library designed for Node.js. This module allows you to train a neural network with a single hidden layer and make predictions based on the trained data.

## Features

- **Pure JavaScript**: No dependencies or external libraries needed.
- **Feedforward Neural Network**: Supports networks with one hidden layer.
- **Backpropagation**: Includes backpropagation for training.
- **Flexible Input and Output Sizes**: You can define the number of input nodes, hidden nodes, and output nodes.
- **Customizable Learning Rate**: Set the learning rate for training as needed.
  
## Installation

You can install Neuradriz using npm.

```bash
npm install neuradriz

Usage

Here is how you can use the Neuradriz neural network module:

## 1. Import the Module

Import the neural network class from the package.

```
const NeuralNetwork = require('neuradriz');
```

## 2. Initialize the Neural Network

Create an instance of the NeuralNetwork by specifying the number of input nodes, hidden nodes, output nodes, and the learning rate.

```
let nn = new NeuralNetwork(4, 16, 1, 0.125);
```

4: Number of input nodes

16: Number of hidden layer nodes

1: Number of output nodes

0.125: Learning rate


## 3. Prepare Training Data

The training data consists of input values and corresponding target values. Below is an example of the data format:

```
const trainingData = [
   {
      input: [5.1, 3.5, 1.4, 0.2],
      target: [0.333]
   },
   {
      input: [5.7, 2.8, 4.1, 1.3],
      target: [0.666]
   }
];
```

Each input is an array of feature values, and each target is an array containing the target output for those input features.

## 4. Train the Neural Network

Train the neural network using the train method. Specify the training data and the number of epochs for which the network will be trained.

```
nn.train(trainingData, 1000);
```

In this example, the network is trained over 1000 epochs.

## 5. Make Predictions

After training, you can use the feedforward method to make predictions.

```
let input = [5.8, 2.7, 4.1, 1.0];
let result = nn.feedforward(input);

console.log(result);  // Output will be the network's prediction
```

## 6. Example

Here’s a full example putting everything together:

```
const NeuralNetwork = require('neuradriz');

// Initialize the neural network
let nn = new NeuralNetwork(4, 16, 1, 0.125);

// Define the training data
const trainingData = [
   { input: [5.1, 3.5, 1.4, 0.2], target: [0.333] },
   { input: [5.7, 2.8, 4.1, 1.3], target: [0.666] },
   { input: [6.3, 3.3, 6.0, 2.5], target: [0.999] }
];

// Train the network
nn.train(trainingData, 1000);

// Test with a new input
let input = [5.8, 2.7, 4.1, 1.0];
let result = nn.feedforward(input);

console.log(result);  // Prediction based on the trained model
```

How It Works

The module works by initializing weights and biases for both the input-to-hidden layer and hidden-to-output layer. It uses the sigmoid activation function for non-linearity and trains using backpropagation with gradient descent.

Feedforward: Computes the output by multiplying inputs by the weights, adding biases, and applying the sigmoid activation.

Backpropagation: Calculates errors and updates weights and biases based on the gradients and the learning rate.



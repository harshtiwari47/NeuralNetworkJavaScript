# Neuradriz 

This project is a fully customizable neural network built using pure JavaScript (ES6). It allows you to create, train, and save a neural network model for various use cases like classification and regression. With support for multiple layers, activation functions, and customizable training options, it offers flexibility and ease of use.

## Features

**Customizable Layers**: Dynamically define input, hidden, and output layers.

**Activation Functions**: Supports Sigmoid and ReLU activation functions.

**Feedforward and Backpropagation**: Implements forward propagation to compute the output and backpropagation for updating weights and biases during training.

**Training Configurations**: Options for batch processing, early stopping, learning rate decay, and validation.

**Save and Load Model:** Save the model state and reload it later for resuming training or making predictions.


## Rules and Constraints

All Layers Must Be Defined: You must define an input layer, at least one hidden layer, and an output layer before training.

***Training & Validation Data Structure: Data must follow the format:***

```json
[
   {
      input: [],  // Array of input values
      target: []  // Array of target values
   }
]
```


## Getting Started

### Installation

You can install Neuradriz using npm.

```bash
npm install neuradriz
```

### 1. Initialize the Neural Network

```javascript
const NeuralNetwork = require('neuradriz');
const nn = new NeuralNetwork();
```

### 2. Define Layers

Define the input, hidden, and output layers. You must have at least one input layer, one hidden layer, and one output layer.

```javascript
// Input Layer (2 nodes, Sigmoid activation)
nn.initLayer({
   type: 'input',
   nodes: 2,
   activation: 'sigmoid'
});

// Hidden Layer (1 node, ReLU activation)
nn.initLayer({
   type: 'hidden',
   nodes: 1,
   activation: 'relu'
});

// Output Layer (1 node, Sigmoid activation)
nn.initLayer({
   type: 'output',
   nodes: 1,
   activation: 'sigmoid'
});
```

### 3. Prepare Training and Validation Data

The data should follow a specific format where each data point has an input and target array.

```javascript
// Training data
const trainingData = [
   { input: [0, 0], target: [0] },
   { input: [0, 1], target: [1] },
   { input: [1, 0], target: [1] },
   { input: [1, 1], target: [1] }
];

// Validation data
const validationDataSet = [
   { input: [0, 1], target: [1] },
   { input: [1, 0], target: [1] },
   { input: [1, 1], target: [1] }
];
```

### 4. Set Learning Rate

You can control how quickly the network learns by adjusting the learning rate.

```javascript
nn.setlearningRate(1.25);
```

### 5. Define Callback for Training

A callback function can track the progress during training. You can use it to log events like the start/end of training, epochs, and batches.

```javascript
async function myCallback({ event, epoch, averageLoss, accuracy }) {
   switch (event) {
      case 'trainStart':
         console.log('Training started.');
         break;
      case 'epochStart':
         console.log(`Epoch ${epoch + 1} started.`);
         break;
      case 'epochEnd':
         console.log(`Epoch ${epoch + 1} ended. Validation Loss: ${averageLoss}, Accuracy: ${accuracy}`);
         break;
      case 'trainEnd':
         console.log('Training completed.');
         break;
      default:
         console.log('Unknown event.');
   }
}
```

### 6. Train the Network

To train the network, you need to provide training data, configure training options like the number of epochs, batch size, validation data, and whether to use early stopping.

```javascript
nn.train(trainingData, {
   epochs: 400,
   batchSize: 10,
   shuffle: true,
   earlyStopping: true,
   patience: 50,
   validationData: validationDataSet,
   lossType: "mse", // Options: "mse", "mae", "crossentropy"
   l2Lambda: 0.01,  // L2 regularization strength
   callback: myCallback // Callback function for tracking training
}).then(() => {
   console.log('Training completed!');
   
   // Save the model after training
   nn.saveModel('trainedModel.json');
   
   // Make a prediction using the trained model
   console.log('Prediction for [1, 1]:', nn.predict([1, 1]));

   // Reload the saved model and continue predictions
   nn.loadModel('trainedModel.json');
   console.log('Reloaded model prediction for [1, 0]:', nn.predict([1, 0]));
});
```

### 7. Making Predictions

To make predictions after training, use the feedforward() function. This will pass input through the network and give you the output.

```javascript
const result = nn.predict([1, 0]);
console.log(`Prediction for [1, 0]: ${result}`);
```

### 8. Saving and Loading the Model

You can save the model state (layers, weights, biases) at any time during or after training, and load it later to continue training or make predictions.

#### Save the Model State

```javascript
nn.saveModel('trainedModel.json');
```

This saves the model's current state to trainedModel.json.

#### Load the Model State

```javascript
nn.loadModel('trainedModel.json');
``` 
This loads the model from the file and restores its configuration and weights, allowing you to resume training or make predictions.

## Training Options (Defaults)

Here’s a summary of all the configurable options during training:

**epochs (required)**: Number of iterations to train the network.

**batchSize (default: 1)**: Number of training samples per batch.

**shuffle (default: false)**: Shuffle the training data at the start of each epoch.

**earlyStopping (default: false)**: Stop training early if validation loss doesn’t improve after a set number of epochs.

**patience (default: 5)**: Number of epochs to wait before stopping early.

**validationData (default: null)**: Data used to validate the model during training.

**lossType (default: "mse")**: The loss function to use. Options are "mse", "mae", or "crossentropy".

**l2Lambda (default: 0.01)**: Strength of L2 regularization to avoid overfitting.

**callback (default: null)**: A function to handle events during training.


## License

This project is licensed under the MIT License.

[Contribution Guidelines](https://github.com/harshtiwari47/neuradriz/blob/main/contribution.md)



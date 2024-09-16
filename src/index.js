class NeuralNetwork {
   constructor() {
      /**
      * Initializes a neural network with default values.
      */
      // Number of input nodes. Default is 1.
      this.inputNodes = 1;

      // Array to hold the configuration of hidden layers.
      this.hiddenLayers = [];

      // Number of output nodes. Default is 1.
      this.outputNodes = 1;

      // Array to track if layers are defined.
      // Index 0: Input layer, 1: Hidden layers, 2: Output layer
      this.definedLayers = [false,
         false,
         false];

      // Learning rate for updating weights during training. Default is 0.1.
      this.learningRate = 0.1;

      // Arrays to store weights and biases for hidden and output layers.
      this.weightsHiddens = [];
      this.biasHidden = [];
      this.biasOutput = [];

      // Array to store activation functions and their derivatives.
      // Activation[0]: Sigmoid, Activation[1]: ReLU
      this.activation = [];

      // Stores the type of activation function used in the output layer.
      this.activationOutput = 0;

      /**
      * Sigmoid activation function and its derivative.
      *
      * @param {number} x - Input value.
      * @returns {number} - Output of the sigmoid function.
      */
      this.activation[0] = {
         ftn(x) {
            return 1 / (1 + Math.exp(-x)); // Sigmoid function
         },
         /**
         * Derivative of the sigmoid function.
         *
         * @param {number} x - Output of the sigmoid function.
         * @returns {number} - Derivative of the sigmoid function.
         */
         ftnDerivative(x) {
            return x * (1 - x); // Derivative of the sigmoid function
         }
      };

      /**
      * ReLU (Rectified Linear Unit) activation function and its derivative.
      *
      * @param {number} x - Input value.
      * @returns {number} - Output of the ReLU function.
      */
      this.activation[1] = {
         ftn(x) {
            return Math.max(0, x); // ReLU function
         },
         /**
         * Derivative of the ReLU function.
         *
         * @param {number} x - Input value.
         * @returns {number} - Derivative of the ReLU function.
         */
         ftnDerivative(x) {
            return x > 0 ? 1: 0; // Derivative of ReLU function
         }
      };

      // Error handling to ensure proper configuration
      if (this.activation.length === 0) {
         console.error('No activation functions defined. Please define at least one activation function.');
      }

      if (this.inputNodes <= 0 || this.outputNodes <= 0) {
         console.error('Input nodes and output nodes should be greater than 0.');
      }

      if (this.learningRate <= 0) {
         console.error('Learning rate should be a positive number.');
      }
   }

   /**
   * Sets the learning rate for the neural network.
   *
   * @param {number} value - The new learning rate to set.
   * @returns {string|undefined} - Returns an error message if the input is invalid; otherwise, it returns nothing.
   */
   setlearningRate(value) {
      if (typeof value === "number" && value > 0) {
         this.learningRate = value;
      } else {
         return "Please provide a valid positive number for Learning Rate!";
      }
   }

   /**
   * Initializes a layer in the neural network based on the provided configuration.
   *
   * @param {Object} config - Configuration object for the layer.
   * @param {string} config.type - Type of the layer: 'input', 'hidden', or 'output'.
   * @param {number} config.nodes - Number of nodes in the layer.
   * @param {string} [config.activation='sigmoid'] - Activation function name: 'sigmoid' or 'relu'. Defaults to 'sigmoid'.
   * @returns {string|void} - Returns an error message if there is an issue with the configuration; otherwise, it performs initialization.
   */
   initLayer(config) {
      // Check if config is an object
      if (typeof config !== "object" || config === null) {
         return "Please provide a valid configuration object!";
      }

      // Ensure the layer type is provided
      if (config.type === undefined) {
         return "Please provide layer type! (input/hidden/output)";
      }

      // Set default activation function if not provided
      if (config.activation === undefined) {
         config.activation = "sigmoid";
      }

      // Validate the activation function
      if (typeof config.activation !== "string" || !["sigmoid", "relu"].includes(config.activation.toLowerCase())) {
         return "Please provide a valid activation name! (sigmoid, relu)";
      }

      // Determine the activation function index
      let activationFtn = config.activation.toLowerCase() === "sigmoid" ? 0: 1;

      // Handle different layer types
      switch (config.type.toLowerCase()) {
         case "hidden":
            this.definedLayers[1] = true;
            this.hiddenLayers.push({
               nodes: config.nodes,
               activation: activationFtn
            });
            break;

         case "input":
            this.inputNodes = config.nodes;
            this.definedLayers[0] = true;
            break;

         case "output":
            this.definedLayers[2] = true;
            this.outputNodes = config.nodes;
            this.biasOutput = this.initializeBias(this.outputNodes);
            this.activationOutput = activationFtn;

            // Initialize weights between hidden layers and output layer
            for (let i = 0; i < this.hiddenLayers.length; i++) {
               if (i === 0) {
                  this.weightsHiddens[i] = this.initializeWeights(this.inputNodes, this.hiddenLayers[i].nodes);
               } else {
                  this.weightsHiddens[i] = this.initializeWeights(this.hiddenLayers[i - 1].nodes, this.hiddenLayers[i].nodes);
               }
            }

            // Initialize weights for the final layer to output nodes
            if (this.definedLayers[1]) {
               this.weightsHiddens[this.weightsHiddens.length] = this.initializeWeights(this.hiddenLayers[this.hiddenLayers.length - 1].nodes, this.outputNodes);
            } else {
               return console.error('No hidden layers found before output layer!');
            }

            // Initialize biases for hidden layers
            for (let i = 0; i < this.hiddenLayers.length; i++) {
               this.biasHidden[i] = this.initializeBias(this.hiddenLayers[i].nodes);
            }
            break;

         default:
            return "Invalid layer type! (input/hidden/output)";
         }
      }

      /**
      * Initializes a weight matrix using Xavier initialization for the given dimensions (rows and columns).
      * Xavier initialization helps in maintaining the variance of activations across layers, improving convergence during training.
      *
      * @param {number} rows - The number of rows (input nodes or neurons from the previous layer).
      * @param {number} cols - The number of columns (output nodes or neurons in the current layer).
      * @returns {number[][]} - A 2D array (matrix) where each value is initialized with random values within a range.
      */
      initializeWeights(rows, cols) {
         // Initialize an empty array to store the weights
         let weights = [];

         // Xavier initialization variance: Helps to scale weights based on the number of input nodes
         let variance = 1 / Math.sqrt(rows); // The standard deviation of the weight distribution

         // Iterate over rows (number of input nodes or previous layer's neurons)
         for (let i = 0; i < rows; i++) {
            weights[i] = [];

            // Iterate over columns (number of output nodes or current layer's neurons)
            for (let j = 0; j < cols; j++) {
               // Generate random weights in the range [-variance, variance]
               weights[i][j] = Math.random() * variance * 2 - variance;
            }
         }

         // Return the 2D weight matrix
         return weights;
      }

      /**
      * Initializes a bias array with small random values for the given size.
      * Bias values help shift the activation function, and initializing them with small values can improve network training.
      *
      * @param {number} size - The number of biases to initialize (usually equal to the number of neurons in a layer).
      * @returns {number[]} - An array of initialized biases with random values.
      */
      initializeBias(size) {
         // Initialize an empty array to store the biases
         let bias = [];

         // Loop through the number of biases to initialize
         for (let i = 0; i < size; i++) {
            // Assign small random values (between 0 and 0.1) to the bias array
            bias[i] = Math.random() * 0.1; // Small random values to avoid large shifts in activation functions initially
         }

         // Return the bias array
         return bias;
      }

      /**
      * Performs matrix multiplication of two 2D arrays (matrices).
      * This function multiplies matrix 'a' with matrix 'b' using the dot product.
      *
      * @param {number[][]} a - The first matrix (left matrix), of size m x n.
      * @param {number[][]} b - The second matrix (right matrix), of size n x p.
      * @returns {number[][]} - The resulting matrix of size m x p after multiplication.
      *
      * Example:
      * If 'a' is 3x2 and 'b' is 2x4, the resulting matrix will be 3x4.
      */
      matrixMultiply(a, b) {
         // Initialize an empty result matrix
         let result = [];

         // Error checking: Ensure the number of columns in 'a' matches the number of rows in 'b'
         if (a[0].length !== b.length) {
            throw new Error("Number of columns in matrix 'a' must equal number of rows in matrix 'b' for multiplication.");
         }

         // Iterate through each row of 'a'
         for (let i = 0; i < a.length; i++) {
            result[i] = []; // Initialize the row in the result matrix

            // Iterate through each column of 'b'
            for (let j = 0; j < b[0].length; j++) {
               result[i][j] = 0; // Initialize the element in the result matrix

               // Perform the dot product of row 'i' from 'a' and column 'j' from 'b'
               for (let k = 0; k < a[0].length; k++) {
                  result[i][j] += a[i][k] * b[k][j];
               }
            }
         }

         return result; // Return the resulting matrix
      }

      /**
      * Adds bias values to each row of the input matrix.
      *
      * @param {number[][]} matrix - The matrix to which the bias values will be added. Each row represents a node layer's outputs.
      * @param {number[]} bias - The bias values to be added to each row. Each bias corresponds to a node in the output layer.
      * @returns {number[][]} - The matrix with biases added to each row.
      */
      addBias(matrix, bias) {
         // Check if matrix is an array and bias is an array of correct length
         if (!Array.isArray(matrix) || !Array.isArray(bias)) {
            console.error("Invalid input: both matrix and bias must be arrays.");
            return matrix;
         }

         // Ensure each row in the matrix has the same number of elements as the bias array
         return matrix.map((row, i) => {
            if (row.length !== bias.length) {
               console.error(`Row ${i} length does not match bias length.`);
               return row; // If the row length doesn't match the bias length, return the row unchanged
            }

            // Add bias to each element of the row
            return row.map((val, j) => val + bias[j]);
         });
      }

      /**
      * Applies the specified activation function to each element in the matrix.
      *
      * @param {number[][]} matrix - A 2D array (matrix) of numbers to apply the activation function to.
      * @param {number} type - The index of the activation function to apply. Default is 0 (e.g., sigmoid).
      *                        It expects that `this.activation` contains activation functions like sigmoid (index 0) and ReLU (index 1).
      * @returns {number[][]} - A new matrix with the activation function applied to each element.
      */
      applyActivation(matrix,
         type = 0) {
         // Check if the type is a valid activation function index
         if (!this.activation[type]) {
            console.error(`Invalid activation type: ${type}. Defaulting to sigmoid (type 0).`);
            type = 0; // Default to sigmoid if an invalid type is provided
         }

         // Ensure the input is a valid matrix (array of arrays)
         if (!Array.isArray(matrix) || !matrix.every(row => Array.isArray(row))) {
            console.error("Invalid matrix format. The input must be a 2D array.");
            return [];
         }

         // Apply the selected activation function to each element in the matrix
         return matrix.map(row => row.map(this.activation[type].ftn.bind(this)));
      }

      /**
      * Calculates the error between the target values and the actual output values.
      *
      * @param {number[]} target - Array of target values (desired outputs).
      * @param {number[]} output - Array of actual output values from the neural network.
      * @returns {number[]} - Array of error values for each output node. If the lengths of the target and output arrays do not match, returns an array filled with zeros.
      */
      calculateError(target, output) {
         // Check if the lengths of target and output arrays match
         if (target.length !== output.length) {
            console.error("Target and output arrays must have the same length.");
            return Array(target.length).fill(0); // Return an array filled with zeros
         }

         // Initialize an array to store the error values
         let error = [];

         // Calculate the error for each output node
         for (let i = 0; i < target.length; i++) {
            error[i] = output[i] - target[i];
         }

         return error;
      }

      /**
      * Calculates the gradients for the output layer.
      * The gradient is the product of the error and the derivative of the activation function for the given output.
      * Gradients are used during backpropagation to update the weights in the network.
      *
      * @param {number[]} errors - The error values (difference between target and output) for each output node.
      * @param {number[]} outputs - The output values for each node in the output layer.
      * @param {number} [activationType=0] - The type of activation function used (0: sigmoid, 1: ReLU, etc.).
      * @returns {number[]} - An array of gradient values for each output node.
      */
      calculateOutputGradient(errors, outputs, activationType = 0) {
         // Initialize an empty array to store the gradients
         let gradients = [];

         // Loop through each error/output to calculate the gradient
         for (let i = 0; i < errors.length; i++) {
            // The gradient is calculated as error * derivative of activation function at the output
            gradients[i] = errors[i] * this.activation[activationType].ftnDerivative(outputs[i]);
         }

         // Return the calculated gradients
         return gradients;
      }

      /**
      * Updates the weights of a neural network layer using the gradients, inputs, and learning rate.
      *
      * This function adjusts the weights based on the calculated gradients, the input values,
      * and the learning rate, which defines how much to adjust the weights by during backpropagation.
      *
      * @param {number[][]} weights - The current weights matrix to be updated.
      * @param {number[]} gradients - The computed gradients for the weights (i.e., error derivatives).
      * @param {number[][]} inputs - The inputs from the previous layer that contribute to the weight update.
      * @param {number} learningRate - The learning rate controlling the size of the update step.
      * @returns {number[][]} - The updated weights matrix.
      */
      updateWeights(weights, gradients, inputs, learningRate) {
         let updatedWeights = [];

         // Iterate over the rows of the weights matrix (number of nodes in the current layer)
         for (let i = 0; i < weights.length; i++) {
            updatedWeights[i] = [];

            // Iterate over the columns (number of nodes in the next layer)
            for (let j = 0; j < weights[i].length; j++) {
               // Update each weight using the gradient, learning rate, and the input value
               // Formula: weight[i][j] = weight[i][j] - learningRate * gradients[j] * inputs[0][i]
               updatedWeights[i][j] = weights[i][j] - learningRate * gradients[j] * inputs[0][i];
            }
         }

         return updatedWeights;
      }

      /**
      * Updates the bias vector using the gradients and the learning rate.
      *
      * This function adjusts each bias value based on the corresponding gradient
      * and the specified learning rate during backpropagation.
      *
      * @param {number[]} bias - The current bias vector to be updated.
      * @param {number[]} gradients - The computed gradients for the biases (i.e., error derivatives).
      * @param {number} learningRate - The learning rate controlling the size of the update step.
      * @returns {number[]} - The updated bias vector.
      */
      updateBias(bias, gradients, learningRate) {
         // Return a new bias array with updated values
         return bias.map((b, i) => b - learningRate * gradients[i]);
      }

      feedforward(input) {
         // Check if all required layers are defined
         if (this.definedLayers.includes(false)) {
            return console.error("Input/Output/Hidden layer is missing");
         }

         let inputMatrix = [input];
         let currentLayerInput = inputMatrix;

         // Forward pass through all hidden layers
         for (let i = 0; i < this.hiddenLayers.length; i++) {
            let hiddenInput = this.matrixMultiply(currentLayerInput, this.weightsHiddens[i]);
            hiddenInput = this.addBias(hiddenInput, this.biasHidden[i]);
            currentLayerInput = this.applyActivation(hiddenInput, this.hiddenLayers[i].activation);
         }

         // Forward pass to the output layer
         let outputInput = this.matrixMultiply(currentLayerInput, this.weightsHiddens[this.weightsHiddens.length - 1]);
         outputInput = this.addBias(outputInput, this.biasOutput);
         let output = this.applyActivation(outputInput, this.activationOutput);

         return output[0]; // Assuming single output neuron
      }

      backpropagate(input, target, newLearningRate = 0.1) {
         // Forward pass: calculate the output and save intermediate hidden outputs
         let inputMatrix = [input]; // Convert input into a matrix for easier manipulation
         let output = this.feedforward(input); // Get network output
         let error = this.calculateError(target, output); // Calculate error
         let outputGradients = this.calculateOutputGradient(error, output); // Compute gradients at the output layer
         let hiddenOutputs = []; // To store the outputs of each hidden layer
         let previousLayerOutput = inputMatrix; // Initially, the input matrix

         // Forward propagation through the hidden layers
         for (let i = 0; i < this.hiddenLayers.length; i++) {
            let hiddenInput = this.matrixMultiply(previousLayerOutput, this.weightsHiddens[i]); // Input to current hidden layer
            hiddenInput = this.addBias(hiddenInput, this.biasHidden[i]); // Add bias
            previousLayerOutput = this.applyActivation(hiddenInput, this.hiddenLayers[i].activation); // Apply activation function
            hiddenOutputs.push(previousLayerOutput); // Store current hidden layer output
         }

         let nextLayerGradients = outputGradients; // Start with gradients from the output layer

         // Backpropagate through the hidden layers, adjusting weights and biases
         for (let i = this.hiddenLayers.length - 1; i >= 0; i--) {
            let currentLayerOutputs = hiddenOutputs[i]; // Outputs of the current hidden layer
            let currentLayerWeights = this.weightsHiddens[i + 1] || []; // Weights connecting to the next layer (output layer if last)
            let currentLayerGradients = [];

            // Calculate gradients for the current hidden layer
            for (let j = 0; j < this.hiddenLayers[i].nodes; j++) {
               let activationType = this.hiddenLayers[i].activation;
               let sum = 0;

               // Calculate the gradient for each node by summing the product of the next layer's gradients and weights
               for (let k = 0; k < nextLayerGradients.length; k++) {
                  sum += nextLayerGradients[k] * currentLayerWeights[j][k];
               }
               // Derivative of the activation function applied to the current layer's outputs
               currentLayerGradients[j] = sum * this.activation[activationType].ftnDerivative(currentLayerOutputs[0][j]);
            }

            // Determine the input to the current layer (input matrix if it's the first hidden layer)
            let prevLayerOutput = i === 0 ? inputMatrix: hiddenOutputs[i - 1];

            // Update weights and biases for the current layer
            this.weightsHiddens[i] = this.updateWeights(this.weightsHiddens[i], currentLayerGradients, prevLayerOutput, newLearningRate);
            this.biasHidden[i] = this.updateBias(this.biasHidden[i], currentLayerGradients, newLearningRate);

            // Pass gradients to the next layer (for the next iteration in backpropagation)
            nextLayerGradients = currentLayerGradients;
         }

         // Update the weights and biases for the output layer
         let lastHiddenOutput = hiddenOutputs[hiddenOutputs.length - 1]; // Output from the last hidden layer
         this.weightsHiddens[this.weightsHiddens.length - 1] = this.updateWeights(this.weightsHiddens[this.weightsHiddens.length - 1], outputGradients, lastHiddenOutput, newLearningRate);
         this.biasOutput = this.updateBias(this.biasOutput, outputGradients, newLearningRate);
      }

      /**
      * Performs prediction by feeding the input through the neural network
      * and applying the activation function to the output.
      *
      * This function uses the trained weights and biases to calculate the network's
      * output for a given input. The final output is processed through the specified
      * activation function (e.g., sigmoid or ReLU).
      *
      * @param {number[]} input - The input values to the neural network.
      * @returns {number[]} - The activated output of the network, representing the prediction.
      */
      predict(input) {
         // Perform feedforward calculation to get raw output (before applying activation)
         const rawOutput = this.feedforward(input);
         return rawOutput
      }

      /**
      * Trains the neural network using the provided training data and configuration.
      *
      * @param {Array} trainingData - The dataset used for training. Each element should be an object with 'input' and 'target' properties.
      * @param {Object} options - Configuration options for training.
      * @param {number} options.epochs - The number of epochs to train for.
      * @param {number} [options.batchSize=1] - The size of each training batch.
      * @param {number} [options.yieldEvery=10] - The number of iterations after which to yield control for async operations.
      * @param {boolean} [options.shuffle=false] - Whether to shuffle the training data before each epoch.
      * @param {boolean} [options.decayLearningRate=false] - Whether to decay the learning rate over epochs.
      * @param {number} [options.decayRate=0.001] - The rate at which to decay the learning rate.
      * @param {boolean} [options.earlyStopping=false] - Whether to apply early stopping based on validation loss.
      * @param {number} [options.patience=5] - The number of epochs to wait for improvement before stopping early.
      * @param {Array} [options.validationData=null] - The dataset used for validation. Each element should be an object with 'input' and 'target' properties.
      * @param {string} [options.lossType='mse'] - The type of loss function to use ('mse', 'mae', or 'crossentropy').
      * @param {number} [options.l2Lambda=0.01] - The regularization strength for L2 regularization.
      * @param {Function} [options.callback=null] - An optional callback function to be called at various stages of training.
      *
      * @returns {Promise<void>}
      */
      async train(trainingData, {
         epochs,
         batchSize = 1,
         yieldEvery = 10,
         shuffle = false,
         decayLearningRate = false,
         decayRate = 0.001,
         earlyStopping = false,
         patience = 5,
         validationData = null,
         lossType = "mse",
         l2Lambda = 0.01,
         callback = null
      }) {
         // Check if all layers are defined
         if (this.definedLayers.some(value => value === false)) {
            return console.error("Input/Output/Hidden layer is missing");
         }

         try {
            let currentLearningRate = this.learningRate;
            let bestLoss = Infinity;
            let epochsWithoutImprovement = 0;

            // Notify callback that training has started
            if (callback) {
               await callback( {
                  event: 'trainStart', epochs, batchSize
               });
            }

            for (let epoch = 0; epoch < epochs; epoch++) {
               // Shuffle data if required
               if (shuffle) {
                  trainingData = this.shuffleData(trainingData);
               }

               // Notify callback that the epoch has started
               if (callback) {
                  await callback( {
                     event: 'epochStart', epoch
                  });
               }

               // Process batches
               for (let i = 0; i < trainingData.length; i += batchSize) {
                  const batch = trainingData.slice(i, i + batchSize);

                  try {
                     // Process each item in the batch
                     for (const {
                        input, target
                     } of batch) {
                        if (callback) {
                           await callback( {
                              event: 'batchStart', input, target
                           });
                        }
                        this.backpropagate(input, target, currentLearningRate);

                        if (callback) {
                           await callback( {
                              event: 'batchEnd', input, target
                           });
                        }
                     }

                  } catch (batchError) {
                     console.error('Error processing batch:', batchError);
                     // Optionally: skip this batch or continue based on the use case
                  }

                  // Apply L2 Regularization
                  this.weightsHiddens = this.weightsHiddens.map(weights => this.applyL2Regularization(weights, l2Lambda));

                  // Yield control for async operations if required
                  if ((i / batchSize) % yieldEvery === 0) {
                     await new Promise(resolve => setTimeout(resolve, 0));
                  }
               }

               // Validation
               if (validationData) {
                  try {
                     const {
                        averageLoss,
                        accuracy
                     } = await this.evaluate(validationData, lossType);

                     // Notify callback that the epoch has ended
                     if (callback) {
                        await callback( {
                           event: 'epochEnd', epoch, averageLoss, accuracy
                        });
                     }

                     // Check for early stopping
                     if (averageLoss < bestLoss) {
                        bestLoss = averageLoss;
                        epochsWithoutImprovement = 0;
                     } else {
                        epochsWithoutImprovement++;
                     }

                     if (epochsWithoutImprovement >= patience && earlyStopping) {
                        console.log(`Early stopping triggered after ${epoch + 1} epochs.`);
                        break;
                     }

                  } catch (evalError) {
                     console.error('Error during evaluation:', evalError);
                     // Optionally: continue training or handle error based on the use case
                  }
               }

               // Update learning rate if decay is enabled
               if (decayLearningRate) {
                  currentLearningRate = this.updateLearningRate(currentLearningRate, epoch, decayRate);
               }
            }

            // Notify callback that training has ended
            if (callback) {
               await callback( {
                  event: 'trainEnd'
               });
            }
         } catch (error) {
            console.error('Error during training:', error);
         }
      }

      /**
      * Evaluates the model's performance on the provided validation data.
      *
      * Calculates the average loss and accuracy based on the specified loss type.
      *
      * @param {Array<Object>} validationData - An array of validation data objects, each containing `input` and `target` properties.
      * @param {string} lossType - The type of loss to use for evaluation. Supported values: 'mse', 'mae', 'crossentropy'.
      * @returns {Promise<Object>} - An object containing the average loss and accuracy.
      *
      * @throws {Error} - Throws an error if lossType is unsupported.
      */
      async evaluate(validationData, lossType) {
         let totalLoss = 0;
         let correctPredictions = 0;
         const predictions = [];
         const targets = [];

         // Validate lossType
         const validLossTypes = ['mse',
            'mae',
            'crossentropy'];
         if (!validLossTypes.includes(lossType)) {
            throw new Error(`Unsupported loss type: ${lossType}. Supported types are: ${validLossTypes.join(', ')}`);
         }

         // Process each validation data entry
         for (const {
            input, target
         } of validationData) {
            // Predict output for the given input
            const prediction = await this.predict(input);

            // Calculate loss based on the provided loss type
            const loss = this.calculateLoss(prediction, target, lossType);
            totalLoss += loss;

            // Collect predictions and targets for accuracy calculation
            predictions.push(prediction[0]);
            targets.push(target[0]);

            // Calculate correct predictions
            if (Math.round(prediction[0]) === target[0]) {
               correctPredictions++;
            }
         }

         // Calculate average loss and accuracy
         const averageLoss = totalLoss / validationData.length;
         const accuracy = correctPredictions / validationData.length;

         // Return results
         return {
            averageLoss,
            accuracy
         };
      }

      /**
      * Calculates the loss based on the specified loss type.
      *
      * The function supports Mean Squared Error (MSE), Mean Absolute Error (MAE), and Cross-Entropy Loss.
      *
      * @param {Array<number>} prediction - An array of predicted values or probabilities.
      * @param {Array<number>} target - An array of target values.
      * @param {string} [lossType='mse'] - The type of loss to calculate. Supported values: 'mse', 'mae', 'crossentropy'.
      * @returns {number} - The calculated loss value.
      *
      * @throws {TypeError} - Throws an error if prediction or target are not arrays.
      * @throws {Error} - Throws an error if prediction and target lengths do not match, or if the lossType is unsupported.
      */
      calculateLoss(prediction, target, lossType = 'mse') {
         // Validate input
         if (!Array.isArray(prediction) || !Array.isArray(target)) {
            throw new TypeError("Prediction and target must be arrays.");
         }
         if (prediction.length !== target.length) {
            throw new Error("Prediction and target arrays must have the same length.");
         }

         // Select loss calculation based on lossType
         switch (lossType.toLowerCase()) {
         case 'mse':
            return this.calculateMeanSquaredError(prediction, target);
         case 'mae':
            return this.calculateMeanAbsoluteError(prediction, target);
         case 'crossentropy':
            return this.calculateCrossEntropyLoss(prediction, target);
         default:
            throw new Error('Unsupported loss type. Supported types are: mse, mae, crossentropy.');
         }
      }

      /**
      * Calculates the Mean Squared Error (MSE) between predicted values and target values.
      *
      * MSE is a measure of the average squared difference between predicted and actual values. It is commonly used
      * to assess the performance of regression models.
      *
      * @param {Array<number>} predictions - An array of predicted values.
      * @param {Array<number>} targets - An array of actual target values.
      * @returns {number} - The Mean Squared Error between predictions and targets.
      *
      * @throws {TypeError} - Throws an error if predictions or targets are not arrays.
      * @throws {Error} - Throws an error if predictions and targets have different lengths.
      */
      calculateMeanSquaredError(predictions, targets) {
         // Validate inputs
         if (!Array.isArray(predictions) || !Array.isArray(targets)) {
            throw new TypeError("Predictions and targets must be arrays.");
         }
         if (predictions.length !== targets.length) {
            throw new Error("Predictions and targets must have the same length.");
         }

         // Calculate the Mean Squared Error
         const mse = predictions.reduce((acc, p, i) => acc + Math.pow(p - targets[i], 2), 0) / targets.length;

         return mse;
      }

      /**
      * Calculates the Mean Absolute Error (MAE) between predicted values and target values.
      *
      * MAE is a measure of the average absolute difference between predicted and actual values. It provides a straightforward
      * measure of prediction accuracy, as it computes the average magnitude of errors.
      *
      * @param {Array<number>} predictions - An array of predicted values.
      * @param {Array<number>} targets - An array of actual target values.
      * @returns {number} - The Mean Absolute Error between predictions and targets.
      *
      * @throws {TypeError} - Throws an error if predictions or targets are not arrays.
      * @throws {Error} - Throws an error if predictions and targets have different lengths.
      */
      calculateMeanAbsoluteError(predictions, targets) {
         // Validate inputs
         if (!Array.isArray(predictions) || !Array.isArray(targets)) {
            throw new TypeError("Predictions and targets must be arrays.");
         }
         if (predictions.length !== targets.length) {
            throw new Error("Predictions and targets must have the same length.");
         }

         // Calculate the Mean Absolute Error
         const mae = predictions.reduce((acc, p, i) => acc + Math.abs(p - targets[i]), 0) / targets.length;

         return mae;
      }

      /**
      * Calculates the Cross-Entropy Loss between predicted probabilities and actual target values.
      *
      * Cross-Entropy Loss is used to quantify the difference between two probability distributions -
      * the predicted probabilities and the actual target values. It's commonly used in binary classification problems.
      *
      * @param {Array<number>} predictions - An array of predicted probabilities for the positive class (values between 0 and 1).
      * @param {Array<number>} targets - An array of actual target values (0 or 1) for the positive class.
      * @returns {number} - The Cross-Entropy Loss.
      *
      * @throws {TypeError} - Throws an error if predictions or targets are not arrays.
      * @throws {Error} - Throws an error if predictions and targets have different lengths.
      * @throws {RangeError} - Throws an error if any prediction value is out of range [0, 1].
      */
      calculateCrossEntropyLoss(predictions, targets) {
         // Validate inputs
         if (!Array.isArray(predictions) || !Array.isArray(targets)) {
            throw new TypeError("Predictions and targets must be arrays.");
         }
         if (predictions.length !== targets.length) {
            throw new Error("Predictions and targets must have the same length.");
         }

         // Initialize the loss accumulator
         let loss = 0;
         const epsilon = 1e-15; // Small value to prevent log(0) and avoid numerical issues

         // Calculate the Cross-Entropy Loss
         for (let i = 0; i < predictions.length; i++) {
            // Validate prediction values
            if (predictions[i] < epsilon || predictions[i] > 1 - epsilon) {
               throw new RangeError("Prediction values must be in the range [0, 1].");
            }

            const p = predictions[i];
            const t = targets[i];

            // Accumulate the loss
            loss -= (t * Math.log(p + epsilon)) + ((1 - t) * Math.log(1 - p + epsilon));
         }

         // Return the average loss
         return loss / predictions.length;
      }


      /**
      * Randomly shuffles the elements in the given dataset using the Fisher-Yates algorithm.
      *
      * This method is essential when training a neural network, as shuffling the dataset
      * ensures that the model does not learn in a fixed order, which could lead to
      * overfitting or poor generalization.
      *
      * @param {Array} data - The dataset to be shuffled. It can be an array of input-output pairs.
      * @returns {Array} - The shuffled dataset.
      */
      shuffleData(data) {
         // Loop through the dataset starting from the end to the second element
         for (let i = data.length - 1; i > 0; i--) {
            // Pick a random index from 0 to i
            const j = Math.floor(Math.random() * (i + 1));

            // Swap elements at index i and index j
            [data[i],
               data[j]] = [data[j],
               data[i]];
         }

         // Return the shuffled dataset
         return data;
      }

      /**
      * Updates the learning rate according to an exponential decay schedule.
      *
      * This method adjusts the learning rate over time to help the model converge more effectively.
      * Initially, the learning rate is higher to make significant updates to the model weights.
      * As training progresses, the learning rate is gradually decreased to fine-tune the model
      * and achieve better convergence.
      *
      * @param {number} epoch - The current epoch or iteration number during training.
      * @param {number} decayRate - The rate at which the learning rate decays over time.
      *                             A higher value will cause the learning rate to decrease faster.
      * @returns {number} - The updated learning rate for the given epoch.
      */
      updateLearningRate(epoch, decayRate) {
         // Calculate the new learning rate using the exponential decay formula
         return this.learningRate / (1 + decayRate * epoch);
      }

      /**
      * Applies L2 regularization to the weights.
      *
      * L2 regularization adds a penalty proportional to the sum of the squared weights
      * to the loss function to prevent overfitting and improve generalization of the model.
      * The regularization term helps in constraining the size of the weights.
      *
      * @param {Array<Array<number>>} weights - A 2D array representing the weights matrix of the network.
      * @param {number} lambda - The regularization parameter that controls the strength of the regularization.
      * @returns {Array<Array<number>>} - The updated weights matrix after applying L2 regularization.
      *
      * @throws {TypeError} - Throws an error if weights is not a 2D array or lambda is not a number.
      */
      applyL2Regularization(weights, lambda) {
         // Validate inputs
         if (!Array.isArray(weights) || !weights.every(row => Array.isArray(row))) {
            throw new TypeError("Weights must be a 2D array.");
         }
         if (typeof lambda !== 'number') {
            throw new TypeError("Lambda must be a number.");
         }

         // Apply L2 regularization to each weight
         return weights.map(row =>
            row.map(value => value * (1 - lambda)) // formula for L2 regularization
         );
      }

      /**
      * Calculates the accuracy of predictions compared to the target values.
      *
      * Accuracy is defined as the proportion of correct predictions to the total number of predictions.
      * The function rounds each prediction to the nearest integer before comparing it to the target value.
      *
      * @param {Array<number>} predictions - An array of predicted values, where each value is a numeric prediction.
      * @param {Array<number>} targets - An array of target values (ground truth), where each value is the true label.
      * @returns {number} - The accuracy of the predictions, a value between 0 and 1.
      *
      * @throws {TypeError} - Throws an error if predictions or targets are not arrays, or if their lengths do not match.
      */
      calculateAccuracy(predictions, targets) {
         // Validate inputs
         if (!Array.isArray(predictions) || !Array.isArray(targets)) {
            throw new TypeError("Predictions and targets must be arrays.");
         }
         if (predictions.length !== targets.length) {
            throw new Error("Predictions and targets must have the same length.");
         }

         // Calculate the number of correct predictions
         let correct = 0;
         for (let i = 0; i < predictions.length; i++) {
            if (Math.round(predictions[i]) === targets[i]) {
               correct++;
            }
         }

         // Return the accuracy as a proportion of correct predictions
         return correct / targets.length;
      }

      /**
      * Saves the current model state (weights, biases, and configuration).
      *
      * @param {string} filename - The file name to save the model state.
      */
      saveModel(filename = 'modelData.json') {
         const modelState = {
            inputNodes: this.inputNodes,
            hiddenLayers: this.hiddenLayers,
            outputNodes: this.outputNodes,
            learningRate: this.learningRate,
            weightsHiddens: this.weightsHiddens,
            biasHidden: this.biasHidden,
            biasOutput: this.biasOutput,
            activationOutput: this.activationOutput,
            activation: this.activation
         };

         // Save to local file (in Node.js environment)
         const fs = require('fs');
         fs.writeFileSync(filename, JSON.stringify(modelState));
         console.log(`Model saved to ${filename}`);
      }

      /**
      * Loads a saved model state from a file and initializes the network with the loaded state.
      *
      * @param {string} filename - The file name from which to load the model state.
      */
      loadModel(filename = 'modelData.json') {
         const fs = require('fs');
         const modelState = JSON.parse(fs.readFileSync(filename));

         this.inputNodes = modelState.inputNodes;
         this.hiddenLayers = modelState.hiddenLayers;
         this.outputNodes = modelState.outputNodes;
         this.learningRate = modelState.learningRate;
         this.weightsHiddens = modelState.weightsHiddens;
         this.biasHidden = modelState.biasHidden;
         this.biasOutput = modelState.biasOutput;
         this.activationOutput = modelState.activationOutput;
         this.activation = modelState.activation;

         console.log(`Model loaded from ${filename}`);
      }
   }

   module.exports = NeuralNetwork
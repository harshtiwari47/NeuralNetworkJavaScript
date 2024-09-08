class NeuralNetwork {
      constructor(inputNodes, hiddenNodes, outputNodes, learningRate) {
         this.inputNodes = inputNodes;
         this.hiddenNodes = hiddenNodes;
         this.outputNodes = outputNodes;
         this.learningRate = learningRate;

         this.weightsInputHidden = this.initializeWeights(this.inputNodes, this.hiddenNodes);
         this.weightsHiddenOutput = this.initializeWeights(this.hiddenNodes, this.outputNodes);

         this.biasHidden = this.initializeBias(this.hiddenNodes);
         this.biasOutput = this.initializeBias(this.outputNodes);
      }

      initializeWeights(rows, cols) {
         let weights = [];
         for (let i = 0; i < rows; i++) {
            weights[i] = [];
            for (let j = 0; j < cols; j++) {
               weights[i][j] = Math.random() * 0.1; // Small random values
            }
         }
         return weights;
      }

      initializeBias(size) {
         let bias = [];
         for (let i = 0; i < size; i++) {
            bias[i] = Math.random() * 0.1; // Small random values
         }
         return bias;
      }

      matrixMultiply(a, b) {
         let result = [];
         for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < b[0].length; j++) {
               result[i][j] = 0;
               for (let k = 0; k < a[0].length; k++) {
                  result[i][j] += a[i][k] * b[k][j];
               }
            }
         }
         return result;
      }

      addBias(matrix, bias) {
         return matrix.map((row, i) => row.map((val, j) => val + bias[j]));
      }

      sigmoid(x) {
         return 1 / (1 + Math.exp(-x));
      }

      sigmoidDerivative(output) {
         return output * (1 - output);
      }

      applyActivation(matrix) {
         return matrix.map(row => row.map(this.sigmoid.bind(this)));
      }

      calculateError(target, output) {
         if (target.length !== output.length) return Array(target.length).fill(0);
         let error = [];
         for (let i = 0; i < target.length; i++) {
            error[i] = output[i] - target[i];
         }
         return error;
      }

      calculateOutputGradient(errors, outputs) {
         let gradients = [];
         for (let i = 0; i < errors.length; i++) {
            gradients[i] = errors[i] * this.sigmoidDerivative(outputs[i]);
         }
         return gradients;
      }

      updateWeights(weights, gradients, inputs, learningRate) {
         let updatedWeights = [];
         for (let i = 0; i < weights.length; i++) {
            updatedWeights[i] = [];
            for (let j = 0; j < weights[i].length; j++) {
               updatedWeights[i][j] = weights[i][j] - learningRate * gradients[j] * inputs[0][i];
            }
         }
         return updatedWeights;
      }

      updateBias(bias, gradients, learningRate) {
         return bias.map((b, i) => b - learningRate * gradients[i]);
      }

      feedforward(input) {
         let inputMatrix = [input];

         let hiddenInput = this.matrixMultiply(inputMatrix, this.weightsInputHidden);
         hiddenInput = this.addBias(hiddenInput, this.biasHidden);
         let hiddenOutput = this.applyActivation(hiddenInput);

         let outputInput = this.matrixMultiply(hiddenOutput, this.weightsHiddenOutput);
         outputInput = this.addBias(outputInput, this.biasOutput);
         let output = this.applyActivation(outputInput);

         return output[0];
      }

      backpropagate(input, target) {
         let inputMatrix = [input];
         let output = this.feedforward(input);

         let error = this.calculateError(target, output);
         let outputGradients = this.calculateOutputGradient(error, output);

         let hiddenInput = this.matrixMultiply(inputMatrix, this.weightsInputHidden);
         hiddenInput = this.addBias(hiddenInput, this.biasHidden);
         let hiddenOutput = this.applyActivation(hiddenInput);

         let hiddenGradients = [];
         for (let j = 0; j < this.hiddenNodes; j++) {
            let sum = 0;
            for (let k = 0; k < this.outputNodes; k++) {
               sum += outputGradients[k] * this.weightsHiddenOutput[j][k];
            }
            hiddenGradients[j] = sum * this.sigmoidDerivative(hiddenOutput[0][j]);
         }

this.weightsHiddenOutput = this.updateWeights(this.weightsHiddenOutput, outputGradients, hiddenOutput, this.learningRate);
         this.weightsInputHidden = this.updateWeights(this.weightsInputHidden, hiddenGradients, inputMatrix, this.learningRate);
         this.biasOutput = this.updateBias(this.biasOutput, outputGradients, this.learningRate);
         this.biasHidden = this.updateBias(this.biasHidden, hiddenGradients, this.learningRate);
      }

      train(trainingData, epochs) {
         for (let epoch = 0; epoch < epochs; epoch++) {
              for (let data of trainingData) {
               this.backpropagate(data.input, data.target);
            }
         }
      }
   }

module.exports = NeuralNetwork

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

      applyActivation(matrix) {
         return matrix.map(row => row.map(this.sigmoid));
      }

      feedforward(input) {
         // Convert input to matrix format
         let inputMatrix = [input];

         // Calculate hidden layer activation
         let hiddenInput = this.matrixMultiply(inputMatrix, this.weightsInputHidden);
         hiddenInput = this.addBias(hiddenInput, this.biasHidden);
         let hiddenOutput = this.applyActivation(hiddenInput);

         // Calculate output layer activation
         let outputInput = this.matrixMultiply(hiddenOutput, this.weightsHiddenOutput);
         outputInput = this.addBias(outputInput, this.biasOutput);
         let output = this.applyActivation(outputInput);

         return output[0];
      }
   }

   let nn = new NeuralNetwork(2, 2, 2, 0.01);
   let input = [0.5, 0.8];

   let result = nn.feedforward(input);
   console.log(result);

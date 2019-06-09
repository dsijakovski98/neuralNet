function activateFunc(x) {
  return 1 / (1 + Math.exp(-x));
}

function dActivate(y) {
  return y * (1 - y);
}

class NeuralNetwork {

  constructor(inputs, hiddens, numHiddens, outputs) {

    this.lr = 0.3; // learning rate
    this.inputNodes = inputs; // number of input nodes
    this.hiddenNodes = hiddens; // number of hidden nodes for each hidden layer
    this.hNum = numHiddens; // number of hidden layers
    this.outputNodes = outputs; // number of output nodes
    this.wNum = numHiddens - 1;

    // WEIGHTS
    this.win = new Matrix(hiddens, inputs); // weights inputs -> hidden nodes
    this.win.randomize();
    this.wh = []; // weights between hidden nodes
    for (var i = 0; i < numHiddens - 1; i++) {
      this.wh[i] = new Matrix(hiddens, hiddens);
      this.wh[i].randomize();
    }
    this.wout = new Matrix(outputs, hiddens); // weights hidden nodes -> outputs
    this.wout.randomize();

    // BIAS
    this.hBias = []; // biases for hidden layers
    for (i = 0; i < numHiddens; i++) {
      this.hBias[i] = new Matrix(hiddens, 1);
      this.hBias[i].randomize();
    }
    this.oBias = new Matrix(outputs, 1); // bias for output layer
    this.oBias.randomize();
  }



  // FEEDFORWARD
  predict(inputArr) {
    let inputs = Matrix.fromArray(inputArr);

    // INITIALIZE HIDDEN LAYERS
    let hiddens = [];
    for (var i = 0; i < this.hNum; i++) {
      hiddens[i] = new Matrix(this.hiddenNodes, 1);
    }

    // FF FROM INPUTS TO FIRST HIDDEN LAYER
    hiddens[0] = Matrix.multiply(this.win, inputs);
    hiddens[0].add(this.hBias[0]);
    hiddens[0].map(activateFunc);

    // FF THROUGH ALL HIDDEN LAYERS
    for (i = 1; i < this.hNum; i++) {
      hiddens[i] = Matrix.multiply(this.wh[i - 1], hiddens[i - 1]);
      hiddens[i].add(this.hBias[i - 1]);
      hiddens[i].map(activateFunc);
    }

    // FF FROM LAST HIDDEN LAYER TO OUTPUTS
    let outputs = new Matrix(this.outputNodes, 1);
    outputs = Matrix.multiply(this.wout, hiddens[this.hNum - 1]);
    outputs.add(this.oBias);
    outputs.map(activateFunc);

    return outputs.toArray();
  }




  // BACKPROPAGATION
  train(inputArr, targets) {
    let inputs = Matrix.fromArray(inputArr);
    let answers = Matrix.fromArray(targets);

    let hiddens = [];
    for (var i = 0; i < this.hNum; i++) {
      hiddens[i] = new Matrix(this.hiddenNodes, 1);
    }

    // FF FROM INPUTS TO FIRST HIDDEN LAYER
    hiddens[0] = Matrix.multiply(this.win, inputs);
    hiddens[0].add(this.hBias[0]);
    hiddens[0].map(activateFunc);

    // FF THROUGH ALL HIDDEN LAYERS
    for (i = 1; i < this.hNum; i++) {
      hiddens[i] = Matrix.multiply(this.wh[i - 1], hiddens[i - 1]);
      hiddens[i].add(this.hBias[i - 1]);
      hiddens[i].map(activateFunc);
    }

    // CALCULATE OUTPUT ERRORS
    let outputs = new Matrix(this.outputNodes, 1);
    outputs = Matrix.multiply(this.wout, hiddens[this.hNum - 1]);
    outputs.add(this.oBias);
    outputs.map(activateFunc);
    let outE = Matrix.subtract(answers, outputs);
    //outE = Matrix.multiply(outE, outE);

    let hE = [];
    for (i = 0; i < this.hNum; i++) {
      hE[i] = new Matrix(this.hiddenNodes, 1);
    }

    // CALCULATE HIDDEN ERRORS
    let woT = this.wout.copy();
    hE[this.hNum - 1] = Matrix.multiply(Matrix.transpose(woT), outE);
    for (i = this.hNum - 2; i >= 0; i--) {
      hE[i] = Matrix.multiply(Matrix.transpose(this.wh[i]), hE[i + 1]);
    }

    // CORRECT THE WEIGHTS AND BIASES
    let gradient = outputs.copy();
    gradient.map(dActivate);
    gradient.multiply(outE);
    gradient.multiply(this.lr);
    this.oBias.add(gradient);
    let hidO = hiddens[this.hNum - 1].copy();
    let deltaO = Matrix.multiply(gradient, Matrix.transpose(hidO));
    this.wout.add(deltaO); // output w/b corrected

    for (i = this.hNum - 1; i > 0; i--) {
      let gradientH = hiddens[i].copy();
      gradientH.map(dActivate);
      gradientH.multiply(hE[i]);
      gradientH.multiply(this.lr);
      this.hBias[i].add(gradientH);
      let hidE = hiddens[i - 1].copy();
      let deltaH = Matrix.multiply(gradientH, Matrix.transpose(hidE));
      this.wh[i - 1].add(deltaH);
    } // hidden w/b corrected

    let gradientI = hiddens[0].copy();
    gradientI.map(dActivate);
    gradientI.multiply(hE[i]);
    gradientI.multiply(this.lr);
    this.hBias[0].add(gradientI);
    let inp = inputs.copy();
    let deltaI = Matrix.multiply(gradientI, Matrix.transpose(inp));
    this.win.add(deltaI); // input w/b corrected
  }


}
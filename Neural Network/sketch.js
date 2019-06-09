let inputs;
let targets;
let brain;

function setup() {
  createCanvas(700, 600);

  brain = new NeuralNetwork(2, 3, 2, 1);
  inputs = [];
  inputs[0] = [0, 0];
  inputs[1] = [0, 1];
  inputs[2] = [1, 0];
  inputs[3] = [1, 1];
  targets = [];
  targets[0] = [0];
  targets[1] = [1];
  targets[2] = [1];
  targets[3] = [0];
}

function draw() {
  background(220);
  for (var i = 0; i < 1000; i++) {
    var j = floor(random(0,4));
    brain.train(inputs[j],targets[j]);
  }
  
  console.log(brain.predict([0,1]))
}
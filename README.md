# **Synaptic caching to reduce energy consumption from learning**

Energy efficiency is a major source of evolutionary selection pressure because it is important for survival. Therefore, in the brain, computational mechanisms such as synaptic plasticity are likely to be optimised to minimise energy consumption wherever possible. In this project, I and two other colleagues have reproduced Li and van Rossum's (Li and van Rossum, 2020) model of different forms of plasticity, in which they propose that transient forms of plasticity serve to cache changes in synaptic weights until later consolidation. To simulate neuron connections and the amount of energy they consume, in this project I constructed a multy-layer perceptron, which I calculated changes in milk energy consumption based on changes in weights by loading it into the widely used MNIST database.

The original paper from Li and van Rossum (2020) can be read **[HERE](https://elifesciences.org/articles/50804)**
<br>
The article describing the project can be read **[HERE](https://github.com/nyirobalazs/multilayer_perceptron/blob/main/Synaptic_caching_to_reduce_energy_consumption_from_learning%20(4).pdf)**

## Multilayer perceptron

The single-layer perceptron is suitable for modelling the energy demand of an individual neuron; however, by using a multilayer perceptron we can take this further, exploring at a cell-system level. The network consists of an input, a hidden and an output layer. The number of hidden units in the middle layer is 100 by default, but this can be changed. Sigmoid functions are used for the input layer, and a softmax function is applied for the output layer. The MNIST database, which contains grayscale images of handwritten numbers, is used to train the network. There are 70,000 images, all 28 x 28 pixels. The database is first normalised so each data point will fall between zero and one. Then the images are divided into train and test groups with a ratio of 60,000:10,000. Forward propagation is performed on all training images one-by-one, and then back-propagation is performed after the errors are calculated. The weights are updated by Stochastic Gradient Descent. The weight changes are first written to a transient weight vector and then, in the absence of consolidation, are immediately written to the weight vector that symbolises persistent memory. It is important to note that although the program computes the sum of the initial, transient and persistent weights, these initial values and subsequent weight changes are stored separately throughout.


This code makes the figures 1, 5, 6 and 7
You can set up the required figure(s) at the end of the code

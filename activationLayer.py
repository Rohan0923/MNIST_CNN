from layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activationPrime):
        self.activation = activation
        self.activationPrime = activationPrime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, outputGradient, LearningRate):
        return np.multiply(outputGradient, self.activationPrime(self.input))
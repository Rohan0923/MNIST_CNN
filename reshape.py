import numpy as np
from layer import Layer

class Reshape(Layer):
    def __init__(self, shapeIn, shapeOut):
        self.shapeIn = shapeIn
        self.shapeOut = shapeOut

    def forward(self, input):
        return np.reshape(input, self.shapeOut)
    
    def backward(self, outputGrad, learnRate):
        return np.reshape(outputGrad, self.shapeIn)
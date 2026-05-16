import numpy as np
from layer import Layer
from scipy import signal

class convolutionL(Layer):
    def __init__(self, inputSize, kernelSize, depth):
        inDepth, h, w = inputSize 
        self.depth = depth
        self.inDepth = inDepth
        self.inputSize = inputSize
        
        self.outputShape = (depth, h - kernelSize + 1, w - kernelSize + 1)
        self.kernelShape = (depth, inDepth, kernelSize, kernelSize)
        
        # Scale down kernel weights to prevent explosion
        self.kernel = np.random.randn(*self.kernelShape) * 0.1
        self.biases = np.zeros(self.outputShape)
    
    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.outputShape)
        for i in range(self.depth):
            for j in range(self.inDepth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernel[i, j], "valid")
        self.output += self.biases
        return self.output        

    def backward(self, outGrad, learnRate):
        kernelGrad = np.zeros(self.kernelShape)
        inputGrad = np.zeros(self.inputSize)

        for i in range(self.depth):
            for j in range(self.inDepth):
                kernelGrad[i, j] = signal.correlate2d(self.input[j], outGrad[i], "valid")
                inputGrad[j] += signal.convolve2d(outGrad[i], self.kernel[i, j], "full")
                
        self.kernel -= learnRate * kernelGrad
        self.biases -= learnRate * outGrad
        return inputGrad
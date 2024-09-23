"""
    author - Ananth Sankar
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

xtrain = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

ytrain = np.array([0, 1, 1, 1])

# print(train)


class Perceptron:
    def __init__(self, inputSize=2, learningRate=0.1, epochs=10):
        self.weights = [0] * inputSize
        self.bias = 0
        self.epochs = epochs
        self.learningRate = learningRate
        self.errors = []
        

    def stepFunction(self, x):
        return 1 if x >= 0 else 0
    
    """
        @ is matrix multiplication
        calculate the weighted sum of inputs + bias
        x is a vector of shape (2,) ie 1d array with 2 elements [1, 0] for example corresponding to a single instance of input
        dot product between two vectors squashes the output into a scalar unit, ie a number
        apply a non-linear step function to introduce non linearity into this linear classification model
    """
    def predict(self, x):
        linearOutput = self.weights @ x + self.bias
        return self.stepFunction(linearOutput)
    
    def train(self, xtrain, ytrain):
        
        for i in range(self.epochs):
            print(f"epoch {i + 1}/{self.epochs}")
            errors = 0
            for i in range(len(xtrain)):
                input = xtrain[i]
                y = ytrain[i]
                
                ydash = self.predict(input)
                error = y - ydash
                errors += error ** 2
                
                self.weights += self.learningRate * error * input
                self.bias += self.learningRate * error
            self.errors.append(errors / len(xtrain)) # MSE
            
    def plot_error(self):
        plt.plot(range(1, self.epochs + 1), self.errors)
        plt.title('Mean Squared Error vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.grid(True)
        plt.show()

    
perceptron = Perceptron()
# print([[0, 1]] @ [[1, 0]] + 2)
perceptron.train(xtrain=xtrain, ytrain=ytrain)

perceptron.plot_error()
for i in range(len(xtrain)):
    pred = perceptron.predict(xtrain[i])
    print(f"{xtrain[i]} | predicted = {pred}")


# perceptron.predict([0, 1])

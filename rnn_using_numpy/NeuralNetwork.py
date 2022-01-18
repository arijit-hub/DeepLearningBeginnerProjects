## Implements the entire neural network skeleton ##
import copy
import numpy as np

class NeuralNetwork:

    def __init__(self , optimizer , weights_initializer , bias_initializer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.inp = None
        self.label = None
        self.out = None
        self.phase = None

    @property
    def phase(self):
        return self.__phase

    @phase.setter
    def phase(self , val):
        self.__phase = val

    def forward(self):

        self.inp , self.label = self.data_layer.next()

        x = self.inp.copy()

        for layer in self.layers:
            layer.testing_phase = self.phase
            x = layer.forward(x)

        data_loss = self.loss_layer.forward(x , self.label)

        reg_loss = 0
        if self.optimizer.regularizer is not None:
            reg_loss = self.optimizer.regularizer.norm(data_loss)

        self.out = data_loss + reg_loss

        return self.out

    def backward(self):

        loss_grad = self.loss_layer.backward(self.label)

        for layer in self.layers[::-1]:
            loss_grad = layer.backward(loss_grad)

    def append_layer(self , layer):

        if layer.trainable == True:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)

    def train(self , iterations):
        self.phase = False
        for iter in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self , input_tensor):
        self.phase = True
        x = input_tensor
        for layer in self.layers:
            layer.testing_phase = self.phase
            x = layer.forward(x)

        return x






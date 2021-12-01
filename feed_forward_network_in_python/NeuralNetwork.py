## Implements the entire neural network skeleton ##
import copy

class NeuralNetwork:

    def __init__(self , optimizer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None

        self.inp = None
        self.label = None
        self.out = None

    def forward(self):

        self.inp , self.label = self.data_layer.next()

        x = self.inp.copy()

        for layer in self.layers:
            x = layer.forward(x)

        self.out = self.loss_layer.forward(x , self.label)

        return self.out

    def backward(self):

        loss_grad = self.loss_layer.backward(self.label)

        for layer in self.layers[::-1]:
            loss_grad = layer.backward(loss_grad)

    def append_layer(self , layer):

        if layer.trainable == True:
            layer.optimizer = copy.deepcopy(self.optimizer)

        self.layers.append(layer)

    def train(self , iterations):

        for iter in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self , input_tensor):

        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)

        return x






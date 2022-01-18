import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):

    def __init__(self):
        super().__init__()

        self.output = None

    def forward(self , input_tensor):

        self.output = 1 / (1 + np.exp(-input_tensor))

        return self.output

    def backward(self , error_tensor):

        input_gradient = (self.output * (1 - self.output)) * error_tensor

        return input_gradient
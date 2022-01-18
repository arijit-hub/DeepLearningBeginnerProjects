import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):

    def __init__(self):
        super().__init__()

        self.output = None

    def forward(self , input_tensor):

        self.output = np.tanh(input_tensor)

        return self.output

    def backward(self , error_tensor):

        input_gradient = (1 - np.square(self.output)) * error_tensor

        return input_gradient
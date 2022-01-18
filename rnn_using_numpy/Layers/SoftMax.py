## Implementation of Softmax Layer ##
import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()

        ## Set a gradient class attribute ##
        self.output = None

    def forward(self , input_tensor):

        ## For stability we make the values clip to a max of 0 ##
        input_tensor = input_tensor - np.max(input_tensor)

        self.output = np.exp(input_tensor) / np.sum(np.exp(input_tensor) , axis = 1 , keepdims = True)

        return self.output

    def backward(self , error_tensor):

        # print('Error tensor shape :' , error_tensor.shape)
        # print('Output Tensor shape :' , self.output.shape)
        input_gradient = self.output * (error_tensor - np.sum(error_tensor * self.output , axis = 1).reshape(-1 , 1))

        return input_gradient





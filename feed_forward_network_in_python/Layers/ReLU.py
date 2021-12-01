## Implementation of the ReLU activation layer ##
import numpy as np
from .Base import BaseLayer
class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()

        ## Set a gradient class attribute ##
        self.relu_gradient = None

    def forward(self , input_tensor):

        ## It performs clipping at zero, since, ReLU is max(0,x). ##
        output_tensor = np.clip(input_tensor , a_min = 0.0 , a_max = None)

        self.relu_gradient = output_tensor.copy()

        return output_tensor

    def backward(self , error_tensor):

        ## The differentiation of relu is : 0 for x < 0 and 1 for x > 0. ##
        self.relu_gradient[self.relu_gradient > 0] = 1


        input_gradient = error_tensor * self.relu_gradient

        return input_gradient

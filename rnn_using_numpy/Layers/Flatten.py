## Implements the generic flatten layer ##
import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):

    def __init__(self):

        super().__init__()

        self.input_shape = None

    def forward(self , input_tensor):

        self.input_shape = input_tensor.shape

        #print(self.input_shape)

        reshaped_tensor = input_tensor.reshape(self.input_shape[0] , -1)

        return reshaped_tensor

    def backward(self , error_tensor):

        reshaped_error = np.reshape(error_tensor , self.input_shape)

        return reshaped_error
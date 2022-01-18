import numpy as np
from .Base import BaseLayer


class Dropout(BaseLayer):

    def __init__(self , probability):
        super().__init__()
        self.probability = probability
        self.mask_arr = None

    def forward(self , input_tensor):
        if self.testing_phase == False:
            self.mask_arr = (1 / self.probability) * np.random.choice([0 , 1] ,
                                                             size = input_tensor.shape ,
                                                             p = [1 - self.probability , self.probability])


            output_tensor =  input_tensor * self.mask_arr

        else:
            output_tensor = input_tensor.copy()

        return output_tensor



    def backward(self , error_tensor):

        input_grad = error_tensor * self.mask_arr

        return input_grad
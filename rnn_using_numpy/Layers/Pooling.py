## Implementation of Max-Pooling Layer ##

import numpy as np
import math
from .Base import BaseLayer

class Pooling(BaseLayer):

    def __init__(self , stride_shape , pooling_shape):

        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.mask = None
        self.output_tensor = None

    def forward(self , input_tensor):

        self.input_tensor = input_tensor

        pool_x = self.pooling_shape[0]
        pool_y = self.pooling_shape[1]

        stride_x = 1
        stride_y = 1

        if len(self.stride_shape) == 1:
            stride_x = self.stride_shape
            stride_y = self.stride_shape

        else:
            stride_x , stride_y = self.stride_shape

        shape_x = math.floor((input_tensor.shape[2] - self.pooling_shape[0]) / stride_x) + 1
        shape_y = math.floor((input_tensor.shape[3] - self.pooling_shape[1]) / stride_y) + 1
        self.output_tensor = np.zeros((input_tensor.shape[0] , input_tensor.shape[1] , shape_x , shape_y))
        self.mask = np.zeros(self.output_tensor.shape)

        for data_num in range(input_tensor.shape[0]):
            for channel in range(input_tensor.shape[1]):
                for i , val1 in enumerate(range( 0 , input_tensor.shape[2] , stride_x)):
                    if val1 + pool_x > input_tensor.shape[2]:
                        break

                    for j , val2 in enumerate(range( 0 , input_tensor.shape[3] , stride_y)):
                        if val2 + pool_y > input_tensor.shape[3]:
                            break

                        max_val = np.max(input_tensor[data_num , channel , val1 : val1 + pool_x , val2 : val2 + pool_y])
                        self.output_tensor[data_num, channel, i, j] = max_val

                        self.mask[data_num , channel , i , j] = np.argmax(input_tensor[data_num , channel , val1 : val1 + pool_x , val2 : val2 + pool_y])

        return self.output_tensor


    def backward(self , error_tensor):

        grad_out = np.zeros((self.input_tensor.shape))
        pool_x = self.pooling_shape[0]
        pool_y = self.pooling_shape[1]

        stride_x = 1
        stride_y = 1

        if len(self.stride_shape) == 1:
            stride_x = self.stride_shape
            stride_y = self.stride_shape

        else:
            stride_x, stride_y = self.stride_shape

        for data_num in range(error_tensor.shape[0]):
            for channel in range(self.input_tensor.shape[1]):
                for i, val1 in enumerate(range(0, self.input_tensor.shape[2], stride_x)):
                    if val1 + pool_x > self.input_tensor.shape[2]:
                        break
                    for j , val2 in enumerate(range(0 , self.input_tensor.shape[3] , stride_y)):
                        if val2 + pool_y > self.input_tensor.shape[3]:
                            break

                        error_val = error_tensor[data_num , channel , i , j]
                        error_max_idx = np.unravel_index(int(self.mask[data_num , channel , i , j]) ,
                                                         grad_out[data_num , channel , val1 : val1 + pool_x , val2 : val2 + pool_y].shape)

                        grad_out[data_num , channel , val1 : val1 + pool_x , val2 : val2 + pool_y][error_max_idx] += error_val


        return grad_out




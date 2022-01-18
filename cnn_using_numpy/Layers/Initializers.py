## Sets up the initialization recipes for neural networks. ##

import numpy as np
import math

class Constant:
    '''
    Deduces the Constant initialization scheme.
    '''
    def __init__(self , constant_value = 0.1):
        self.constant_value = constant_value

    def initialize(self , weights_shape , fan_in = None , fan_out = None):

        weights_tensor = np.ones(weights_shape) * self.constant_value
        return weights_tensor

class UniformRandom:
    '''
    Deduces the Uniform Random Distribution based initialization scheme.
    '''

    def __init__(self):
        self.low = 0.0
        self.high = 1.0

    def initialize(self , weights_shape , fan_in = None , fan_out = None):

        weights_tensor = np.random.uniform(low = self.low , high = self.high , size = weights_shape)

        return weights_tensor


class Xavier:
    '''
    Deduces the Xavier or Glorot initialization scheme.
    '''

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in=None, fan_out=None):

        weights_tensor = np.random.normal(0.0 , math.sqrt(2 / (fan_in + fan_out)) , size=weights_shape)

        return weights_tensor




class He:
    '''
    Deduces the He initialization scheme which works amazingly well in conjuction with ReLU
    non-linearity.
    '''

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        weights_tensor = np.random.normal(0.0 , math.sqrt(2 / fan_in) , size=weights_shape)

        return weights_tensor
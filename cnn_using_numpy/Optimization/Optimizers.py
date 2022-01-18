## Python module for optimizers ##

import numpy as np
import math

class Sgd:
    '''
    Implements the Stocastic gradient descent learning algorithm.
    '''
    def __init__(self, learning_rate):
        assert(type(learning_rate) == float or type(learning_rate) == int)
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        ## Perform SGD by changing the weight tensor with gradient and learning rate ##
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight


class SgdWithMomentum:
    '''
    Implements the SGD with Momentum optimization algorithm.
    '''
    def __init__(self , learning_rate , momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prev_v = 0.0


    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.momentum_rate * self.prev_v - self.learning_rate * gradient_tensor
        self.prev_v = v
        updated_weight = weight_tensor + v
        return updated_weight


class Adam:
    '''
    Implements the Adam optimization algorithm.
    '''
    def __init__(self , learning_rate , mu , rho):
        self.learning_rate = learning_rate
        self.k = 1
        self.mu = mu
        self.rho = rho
        self.prev_v = 0.0
        self.prev_r = 0.0

    def calculate_update(self, weight_tensor, gradient_tensor):
        g = gradient_tensor

        v = self.mu * self.prev_v + (1 - self.mu) * g
        self.prev_v = v

        r = self.rho * self.prev_r + (1 - self.rho) * (g ** 2)
        self.prev_r = r

        v = v / (1 - (self.mu ** self.k))
        r = r / (1 - (self.rho ** self.k))

        self.k = self.k + 1

        updated_weight = weight_tensor - self.learning_rate * (v / (np.sqrt(r) + np.finfo(float).eps))

        return updated_weight




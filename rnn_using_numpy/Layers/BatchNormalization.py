import numpy as np
import copy
from .Base import BaseLayer
from .Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):

    def __init__(self , channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.mu = 0.0
        self.sigma_sq = 0.0
        self.weights , self.bias = np.ones(self.channels) , np.zeros(self.channels)
        self.intermediate = None
        self.optimizer = None
        self.current_mean = None
        self.current_variance = None
        self.input_tensor = None
        self.gradient_weights = None
        self.gradient_bias = None

    @property
    def gradient_weights(self):
        ## Getting the gradient weights ##

        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        ## Setting the gradient weight ##

        self.__gradient_weights = val

    @property
    def gradient_bias(self):
        ## Getting the gradient weights ##

        return self.__gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, val):
        ## Setting the gradient weight ##

        self.__gradient_bias = val

    ## Defining our Pythonic optimizer property ##
    @property
    def optimizer(self):
        ## This acts as the getter property for the optimizer attribute ##

        return self.__optimizer

    @optimizer.setter
    def optimizer(self, val):
        ## This acts as the setter property for the optimizer attribute ##
        self.__optimizer = val

    def initialize(self , weights_initializer , bias_initializer):
        fan_in = self.channels
        self.weights = weights_initializer.initialize(self.channels , fan_in)
        self.bias = bias_initializer.initialize(self.channels , fan_in)

    def forward(self , input_tensor):

        self.input_tensor = input_tensor.copy()

        if len(input_tensor.shape) == 4:
            input_tensor = self.reformat(input_tensor)

        if self.testing_phase == False:
            alpha = 0.8

            mean = np.mean(input_tensor , axis=0)
            variance = np.var(input_tensor , axis=0)
            self.current_mean = mean.copy()
            self.current_variance = variance.copy()
            self.intermediate = (input_tensor - mean) / (np.sqrt(variance + np.finfo(float).eps))
            self.mu = alpha * self.mu + (1 - alpha) * mean
            self.sigma_sq = alpha * self.sigma_sq + (1 - alpha) * variance

        else:
            self.intermediate = (input_tensor - self.current_mean) / (np.sqrt(self.current_variance + np.finfo(float).eps))
            #self.intermediate = (input_tensor - self.mu) / (np.sqrt(self.sigma_sq + np.finfo(float).eps))
        output_tensor = self.weights * self.intermediate + self.bias

        if (len(self.input_tensor.shape) == 4):
            output_tensor = self.reformat(output_tensor)

        return output_tensor

    def backward(self , error_tensor):

        if len(self.input_tensor.shape) == 4:
            error_tensor = self.reformat(error_tensor)

        self.gradient_weights = np.sum(error_tensor * self.intermediate , axis=0)
        self.gradient_bias = np.sum(error_tensor , axis=0)

        if self.optimizer != None:
            weights_optimizer = copy.deepcopy(self.optimizer)
            bias_optimizer = copy.deepcopy(self.optimizer)

            self.weights = weights_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        reshaped_inp = self.input_tensor.copy()

        if len(self.input_tensor.shape) == 4:
            reshaped_inp = self.reformat(self.input_tensor)
        grad_input = compute_bn_gradients(error_tensor ,  reshaped_inp,
                                          self.weights , self.current_mean,
                                          self.current_variance)

        if len(self.input_tensor.shape) == 4:
            grad_input = self.reformat(grad_input)

        return grad_input

    def reformat(self , tensor):

        if len(tensor.shape) == 2:
            new_tensor = tensor.reshape(self.input_tensor.shape[0] ,
                                         self.input_tensor.shape[2] * self.input_tensor.shape[3] ,
                                         tensor.shape[1])

            new_tensor = new_tensor.transpose(0 , 2 , 1)

            new_tensor = new_tensor.reshape(new_tensor.shape[0],
                                            new_tensor.shape[1],
                                            self.input_tensor.shape[2],
                                            self.input_tensor.shape[3])
            return new_tensor

        elif len(tensor.shape) == 4:
            new_tensor = tensor.reshape(tensor.shape[0] , tensor.shape[1] , tensor.shape[2] * tensor.shape[3])
            new_tensor = np.transpose(new_tensor , axes = (0 , 2 , 1))
            new_tensor = new_tensor.reshape(-1 , new_tensor.shape[2])
            return new_tensor





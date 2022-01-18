## Implements the fully connected layer ##
import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size , output_size):

        ## Instantiating the parent class constructor ##
        super().__init__()

        ## Setting the input and the output sizes as class variables ##
        self.input_size = input_size
        self.output_size = output_size

        ## Setting the inherited attribute of trainable to True ##
        self.trainable = True

        ## Setting the weights attribute via uniform random distribution ##
        self.weights = np.random.uniform(0.,1.,(self.input_size + 1 , self.output_size))

        ## Setting the optimizer ##
        self.optimizer = None

        ## Setting the gradient weights ##
        self.gradient_weights = None

        ## Setting the input tensor ##
        self.input_tensor = None

    def initialize(self , weights_initializers , bias_initializers):

        fan_in = self.input_size
        fan_out = self.output_size
        self.weights[:-1] = weights_initializers.initialize((self.input_size , self.output_size) , fan_in , fan_out)
        self.weights[-1] = bias_initializers.initialize((1 , self.output_size) , fan_in , fan_out)

    def forward(self , input_tensor):

        ## Getting the result of one forward pass, which is nothing but the linear combination...##
        ## ...of input_tensor and weights ##


        ## Setting the bias vector ##
        ## There should be one bias for each sample of data ##
        bias_list = [1] * input_tensor.shape[0]

        ## Reshaping the bias array to (batch_size,1) to help in concatenation ##
        bias_array = np.array(bias_list).reshape(input_tensor.shape[0] , 1)

        ## Add in the bias term to the input tensor and storing it in the class variable ##
        ## This will be used for the backward pass ##
        self.input_tensor = np.concatenate((input_tensor , bias_array) , axis = 1)

        ## The output of the forward pass is nothing but the dot product of the input and the weights ##
        output_tensor = np.dot(self.input_tensor , self.weights)

        ## Its always better to assert the shape of the output to check if it matches ##
        assert(output_tensor.shape == (self.input_tensor.shape[0] , self.output_size))

        return output_tensor

    ## Defining our Pythonic optimizer property ##
    @property
    def optimizer(self):

        ## This acts as the getter property for the optimizer attribute ##

        return self.__optimizer

    @optimizer.setter
    def optimizer(self, val):

        ## This acts as the setter property for the optimizer attribute ##
        self.__optimizer = val

    def backward(self , error_tensor):

        ## Implements the backward pass or the backpropagation ##

        ## For the usage of backpropagation in the previous layer the gradients of the ...##
        ## inputs to the current layer must be passed on. ##
        ## But there is one catch, we tend to add in the bias term to the input as well as the bias weights...##
        ## ...hence after calculating the weights we must leave out our bias term, which is the last term. ##
        gradient_inputs = np.dot(error_tensor , self.weights.T)[:,:-1]

        ## Now to optimize the paramter of our current layer we must get the gradient of the loss w.r.t the ...##
        ## current layer's parameter ##
        self.gradient_weights = np.dot(self.input_tensor.T , error_tensor)

        self.__gradient_weights = self.gradient_weights

        ## We must optimize our parameter via the optimizer. But this should only be done if we have an optimizer ##
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return gradient_inputs

    ## Defining the setter of our property gradient weights ##
    @property
    def gradient_weights(self):

        ## Getting the gradient weights ##

        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self , val):

        ## Setting the gradient weight ##

        self.__gradient_weights = val








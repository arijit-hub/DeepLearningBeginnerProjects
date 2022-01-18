import numpy as np
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid

class RNN:

    def __init__(self , input_size , hidden_size , output_size):

        self.trainable = True
        self.testing_phase = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_state = np.zeros((1 , hidden_size))
        self.memorize = False

        self.concatenated_inp = None
        self.hidden = None
        self.hid_state = None
        self.intermediate_out = None
        self.output = None
        self.input_tensor = None

        self.fc1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc2 = FullyConnected(self.hidden_size, self.output_size)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        self.fc1.optimizer , self.fc2.optimizer = None , None

        self.weights = np.random.uniform(0., 1., (self.fc1.weights.shape))

        self.gradient_weights = np.zeros((self.fc1.weights.shape))
        self.gradient_weights2 = np.zeros((self.fc2.weights.shape))
        self.optimizer = None

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
        self.fc1.initialize(weights_initializer, bias_initializer)
        self.fc2.initialize(weights_initializer, bias_initializer)


    @property
    def memorize(self):

        return self.__memorize

    @memorize.setter
    def memorize(self , val):

        self.__memorize = val

    @property
    def gradient_weights(self):

        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):

        self.__gradient_weights = val


    @property
    def weights(self):

        return self.fc1.weights

    @weights.setter
    def weights(self, val):

        self.fc1.weights = val


    def forward(self , input_tensor):

        self.input_tensor = input_tensor.copy()
        self.concatenated_inp = np.zeros((input_tensor.shape[0] , int(self.input_size + self.hidden_size)))
        self.hidden = np.zeros((input_tensor.shape[0] , self.hidden_size))
        self.hid_state = np.zeros((input_tensor.shape[0] , self.hidden_size))
        self.intermediate_out = np.zeros((input_tensor.shape[0] , self.output_size))
        self.output = np.zeros((input_tensor.shape[0] , self.output_size))

        if not self.memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))

        for idx , sequences in enumerate(input_tensor):
            self.concatenated_inp[idx] = np.concatenate((sequences.reshape(1 , -1), self.hidden_state) , axis = 1)
            self.hidden[idx] = self.fc1.forward(self.concatenated_inp[idx].reshape(1 , -1))
            self.hidden_state = self.tanh.forward(self.hidden[idx].reshape(1 , -1))
            self.hid_state[idx] = self.hidden_state.copy()
            self.intermediate_out[idx] = self.fc2.forward(self.hidden_state.reshape(1 , -1))
            self.output[idx] = self.sigmoid.forward(self.intermediate_out[idx].reshape(1 , -1))


        return self.output

    def backward(self , error_tensor):

        gradient_inputs = np.zeros(self.input_tensor.shape)
        gradient_hidden = 0
        self.gradient_weights = np.zeros((self.fc1.weights.shape))
        self.gradient_weights2 = np.zeros((self.fc2.weights.shape))

        for idx ,error_sequence in enumerate(reversed(error_tensor)):

            self.sigmoid.output = self.output[len(error_tensor) - (idx + 1)]
            error_sigmoid = self.sigmoid.backward(error_sequence)

            hid_intermediate = self.hid_state[len(error_tensor) - (idx + 1)]
            hid = np.insert(hid_intermediate , -1 , 1)
            self.fc2.input_tensor = hid.reshape(1 , -1)
            error_fc2 = self.fc2.backward(error_sigmoid.reshape(1 , -1))
            self.gradient_weights2 += self.fc2.gradient_weights.copy()

            self.tanh.output = self.hid_state[len(error_tensor) - (idx + 1)]
            half_error_fc2 = error_fc2 + gradient_hidden
            error_tanh = self.tanh.backward(half_error_fc2.reshape(1 , -1))

            inp = self.concatenated_inp[len(error_tensor) - (idx + 1)]
            inp = np.concatenate((inp , np.array([1]))).reshape(1 , -1)
            self.fc1.input_tensor = inp
            error_fc1 = self.fc1.backward(error_tanh.reshape(1 , -1))
            self.gradient_weights = self.gradient_weights + self.fc1.gradient_weights.copy()

            gradient_inputs[len(error_tensor) - (idx + 1)] = error_fc1.reshape(1 , -1)[: , : self.input_size].copy()
            gradient_hidden = error_fc1.reshape(1 , -1)[: , self.input_size : self.input_size + self.hidden_size].copy()

        if self.optimizer is not None:

            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.fc2.weights = self.optimizer.calculate_update(self.fc2.weights , self.gradient_weights2)


        return gradient_inputs
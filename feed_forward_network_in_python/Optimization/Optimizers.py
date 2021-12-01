## Python module for optimizers ##

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
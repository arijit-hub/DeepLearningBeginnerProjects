## Implements the Base layer ##

class BaseLayer:
    def __init__(self , weights = None):
        self.trainable = False
        self.testing_phase = False
        self.weights = weights
## Implements the Base layer ##

class BaseLayer:
    def __init__(self , weights = None):
        self.trainable = False
        self.weights = weights
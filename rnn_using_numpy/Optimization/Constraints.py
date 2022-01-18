import numpy as np


class L2_Regularizer:

    def __init__(self , alpha):
        self.alpha = alpha

    def calculate_gradient(self , weights):
        sub_grad = self.alpha * weights
        return sub_grad

    def norm(self , weights):
        normed_weights = self.alpha * np.sum((np.linalg.norm(weights , keepdims=True) ** 2))
        return normed_weights


class L1_Regularizer:

    def __init__(self , alpha):
        self.alpha = alpha

    def calculate_gradient(self , weights):
        sub_grad = self.alpha * np.sign(weights)
        return sub_grad

    def norm(self, weights):
        normed_weights = self.alpha * np.sum(np.absolute(weights))
        return normed_weights
